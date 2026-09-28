//! A process descriptor table: the mapping from [`RawFd`] slots to shared
//! [`OpenFileDescription`]s, plus the backends the process can open files from.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::ofd::{OpenFileDescription, SharedRead};
use crate::resource::Filesystem;
use crate::types::{Access, Error, FileId, RawFd, ReadOutcome, Result, MAX_FD};

/// One process's view of the file system: an fd table and a registry of named backends.
///
/// Cloning via [`Process::fork`] shares OFDs (POSIX fork semantics); it does **not** copy the
/// resources or their data — only the slot→`Arc<OFD>` map is duplicated, and each entry points at
/// the same `Arc<OpenFileDescription>` as the parent's.
#[derive(Debug)]
pub struct Process {
    pid: u32,
    max_fd: RawFd,
    table: Mutex<Table>,
    registry: Mutex<HashMap<&'static str, Arc<dyn Filesystem>>>,
}

/// The slot→OFD map plus a `next_free` accelerator and an `open_count` occupancy counter.
///
/// `next_free` is a **lower bound** on the lowest free slot: the invariant is
/// `∀ i < next_free, slots[i] is occupied`. So allocation never rescans from 0, and returns the
/// *lowest* free fd (POSIX `open`) — a full table is rejected in O(1) via the `open_count`
/// pigeonhole (see below), not by comparing `next_free` to `max_fd` (that equality only holds
/// for a sequential fill and breaks after a pinned `open_at`). `close` lowers `next_free` to the
/// freed slot, so the next allocation lands there first.
#[derive(Debug, Clone)]
struct Table {
    slots: Vec<Option<Arc<OpenFileDescription>>>,
    next_free: RawFd,
    /// Slots currently occupied. Because every occupied slot is ≤ `max_fd`, `open_count == max_fd + 1`
    /// means the table is provably full — so a full-table `open` is rejected in O(1) without scanning,
    /// even when `next_free` is a stale low bound (e.g. after pinned `open_at` fills).
    open_count: u32,
}

impl Table {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            next_free: 0,
            open_count: 0,
        }
    }

    /// Lowest free slot ≤ `max_fd`, or `None` if the table is full.
    ///
    /// O(1) when full (`open_count` proves it by pigeonhole — no scan). Otherwise a walk that starts
    /// at the `next_free` lower bound and only advances past genuinely-occupied slots, so steady-state
    /// fill and hole-reuse after `close` are both O(1). The residual linear cost is bounded by the
    /// distance from `next_free` to the nearest free slot, which no lowest-free allocator can avoid.
    fn alloc_lowest(&mut self, max_fd: RawFd) -> Option<RawFd> {
        if self.open_count as u64 > max_fd as u64 {
            return None; // every slot 0..=max_fd is occupied
        }
        // Invariant ⇒ every slot < next_free is occupied, so the lowest free is at or above it.
        let mut i = self.next_free;
        while i <= max_fd && matches!(self.slots.get(i as usize), Some(Some(_))) {
            i += 1;
        }
        if i > max_fd {
            return None;
        }
        self.next_free = i + 1;
        Some(i)
    }

    fn place(&mut self, fd: RawFd, ofd: Arc<OpenFileDescription>) {
        let idx = fd as usize;
        if self.slots.len() <= idx {
            self.slots.resize(idx + 1, None);
        }
        self.slots[idx] = Some(ofd);
        self.open_count += 1;
        // next_free is a lower bound, so occupying a slot (>= next_free via alloc_lowest, or any slot
        // via pinned open_at) never invalidates it — it only ever makes the bound weaker, never wrong.
    }

    /// Free a slot, lowering `next_free` if the hole is below it (so the next open reuses it first).
    fn release(&mut self, fd: RawFd) -> bool {
        match self.slots.get_mut(fd as usize) {
            Some(slot @ Some(_)) => {
                *slot = None;
                self.open_count -= 1;
                if fd < self.next_free {
                    self.next_free = fd;
                }
                true
            }
            _ => false,
        }
    }

    fn get(&self, fd: RawFd) -> Option<Arc<OpenFileDescription>> {
        self.slots.get(fd as usize).and_then(|s| s.clone())
    }

    fn is_open(&self, fd: RawFd) -> bool {
        matches!(self.slots.get(fd as usize), Some(Some(_)))
    }
}

impl Process {
    pub fn new(pid: u32) -> Self {
        Self::with_max_fd(pid, MAX_FD)
    }

    /// Build a process with a small descriptor ceiling — used to exercise
    /// [`Error::DescriptorTableFull`] without allocating 64k slots.
    pub fn with_max_fd(pid: u32, max_fd: RawFd) -> Self {
        assert!(max_fd >= 3, "a process needs at least stdio slots");
        Self {
            pid,
            max_fd,
            table: Mutex::new(Table::new()),
            registry: Mutex::new(HashMap::new()),
        }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Register a backend under its own name so `open` and `shutdown` can reach it. Returns the name.
    pub fn register(&self, fs: Arc<dyn Filesystem>) -> &'static str {
        let name = fs.name();
        self.lock_registry().insert(name, fs);
        name
    }

    pub fn backend(&self, name: &str) -> Option<Arc<dyn Filesystem>> {
        self.lock_registry().get(name).cloned()
    }

    // ---- open / close -------------------------------------------------------------

    /// Open `id`, returning the lowest free descriptor (POSIX `open`).
    pub fn open(&self, id: FileId, access: Access) -> Result<RawFd> {
        let ofd = self.open_ofd(id, access)?;
        let mut table = self.lock_table();
        let fd = table
            .alloc_lowest(self.max_fd)
            .ok_or(Error::DescriptorTableFull)?;
        table.place(fd, ofd);
        Ok(fd)
    }

    /// Open `id` **pinned to `fd`**, refusing to clobber a live slot (the journal/checkpoint
    /// contract: restore must be able to reproduce an exact slot number). On success returns
    /// `Ok(fd)`.
    pub fn open_at(&self, fd: RawFd, id: FileId, access: Access) -> Result<RawFd> {
        if fd > self.max_fd {
            return Err(Error::DescriptorTableFull);
        }
        let ofd = self.open_ofd(id, access)?;
        let mut table = self.lock_table();
        if table.is_open(fd) {
            return Err(Error::SlotAlreadyInUse(fd));
        }
        table.place(fd, ofd);
        Ok(fd)
    }

    /// Close a descriptor. This drops **one** reference to the OFD; the resource is freed only
    /// when the last descriptor and any in-flight reader release it. There is no explicit
    /// "handle count" to decrement — `Arc` *is* the refcount. Returns `true` if a slot was freed.
    pub fn close(&self, fd: RawFd) -> Result<bool> {
        let mut table = self.lock_table();
        if table.release(fd) {
            Ok(true)
        } else {
            Err(Error::BadDescriptor(fd))
        }
    }

    // ---- dup / fork ---------------------------------------------------------------

    /// Duplicate a descriptor to the lowest free slot. The new fd shares the *same* OFD, so the
    /// two descriptors advance one cursor together — the same observable behavior as `fork`, and
    /// the cleanest way to demonstrate "separate open ⇒ separate OFD" vs "dup/fork ⇒ shared OFD".
    pub fn dup(&self, fd: RawFd) -> Result<RawFd> {
        let ofd = self.get(fd)?;
        let mut table = self.lock_table();
        let new_fd = table
            .alloc_lowest(self.max_fd)
            .ok_or(Error::DescriptorTableFull)?;
        table.place(new_fd, ofd);
        Ok(new_fd)
    }

    /// POSIX fork: a new process whose descriptor table references the *same* OFDs.
    pub fn fork(&self) -> Process {
        let child = Process {
            pid: self.pid.wrapping_add(1),
            max_fd: self.max_fd,
            table: Mutex::new(self.lock_table().clone()),
            registry: Mutex::new(self.lock_registry().clone()),
        };
        child
    }

    // ---- reads --------------------------------------------------------------------

    /// Sequential read advancing the shared cursor. Blocks behind a contended cursor.
    pub fn read(&self, fd: RawFd, buf: &mut [u8]) -> Result<ReadOutcome> {
        self.get(fd)?.read(buf)
    }

    /// Non-blocking sequential read: [`Error::WouldBlock`] if the cursor is momentarily held.
    pub fn try_read(&self, fd: RawFd, buf: &mut [u8]) -> Result<ReadOutcome> {
        self.get(fd)?.try_read(buf)
    }

    /// Zero-copy sequential read: `Ok(Some(..))` borrows from the resource when the backend can
    /// hand one out, `Ok(None)` when it cannot (caller falls back to [`Process::read`]).
    pub fn read_shared(&self, fd: RawFd, want: usize) -> Result<Option<SharedRead>> {
        self.get(fd)?.read_shared(want)
    }

    /// Positioned read that does not move the shared cursor (POSIX `pread`).
    pub fn read_at(&self, fd: RawFd, offset: u64, buf: &mut [u8]) -> Result<usize> {
        self.get(fd)?.read_at(offset, buf)
    }

    pub fn position(&self, fd: RawFd) -> Result<u64> {
        Ok(self.get(fd)?.position())
    }

    pub fn seek(&self, fd: RawFd, to: u64) -> Result<u64> {
        Ok(self.get(fd)?.seek(to))
    }

    // ---- introspection ------------------------------------------------------------

    /// Resolve a descriptor to its shared OFD, or [`Error::BadDescriptor`].
    pub fn get(&self, fd: RawFd) -> Result<Arc<OpenFileDescription>> {
        self.lock_table().get(fd).ok_or(Error::BadDescriptor(fd))
    }

    pub fn is_open(&self, fd: RawFd) -> bool {
        self.lock_table().is_open(fd)
    }

    // ---- whole file system --------------------------------------------------------

    /// "Close the whole file system": shut a named backend down. Open descriptors stay in the
    /// table (so callers can still observe them) but their next read returns
    /// [`Error::BackendShutdown`]; further opens fail. The prototype does not tear the table down
    /// because doing so is not the backend's to decide — see DESIGN.md.
    pub fn shutdown(&self, backend: &str) -> Result<()> {
        let fs = self
            .backend(backend)
            .ok_or_else(|| Error::UnknownBackend(backend.to_string()))?;
        fs.shutdown()
    }

    // ---- internals ----------------------------------------------------------------

    fn open_ofd(&self, id: FileId, access: Access) -> Result<Arc<OpenFileDescription>> {
        let fs = self.backend(id.backend).ok_or(Error::NoSuchFile(id))?;
        let resource = fs.open(id, access)?;
        Ok(OpenFileDescription::new(access, resource))
    }

    fn lock_table(&self) -> MutexGuard<'_, Table> {
        self.table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_registry(&self) -> MutexGuard<'_, HashMap<&'static str, Arc<dyn Filesystem>>> {
        self.registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
