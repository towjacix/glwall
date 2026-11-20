use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use calloop::generic::Generic;
use calloop::{Interest, LoopHandle, Mode, PostAction};
use inotify::{EventMask, Inotify, WatchDescriptor, WatchMask, Watches};

type WatchMap = Arc<Mutex<HashMap<WatchDescriptor, (PathBuf, HashSet<String>)>>>;

pub struct ShaderWatcher {
    watches: Watches,
    by_dir: HashMap<PathBuf, WatchDescriptor>,
    by_wd: WatchMap,
}

impl ShaderWatcher {
    pub fn insert_into_loop<D, F>(
        handle: &LoopHandle<'static, D>,
        mut on_shader_changed: F,
    ) -> Result<Self>
    where
        D: 'static,
        F: FnMut(&mut D, &Path) + 'static,
    {
        let inotify = Inotify::init().context("failed to initialize inotify")?;
        let watches = inotify.watches();
        let by_wd: WatchMap = Arc::new(Mutex::new(HashMap::new()));
        let cb_map = Arc::clone(&by_wd);

        handle
            .insert_source(
                Generic::new(inotify, Interest::READ, Mode::Level),
                move |_readiness, wrapper, state| {
                    let mut buffer = [0u8; 4096];
                    let mut changed_paths = Vec::new();
                    loop {
                        let events = match unsafe { wrapper.get_mut() }.read_events(&mut buffer) {
                            Ok(events) => events,
                            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                            Err(err) => {
                                tracing::warn!("inotify read error: {err}");
                                break;
                            }
                        };

                        let guard = cb_map.lock().unwrap();
                        for ev in events {
                            if !ev.mask.intersects(
                                EventMask::CLOSE_WRITE | EventMask::MOVED_TO | EventMask::CREATE,
                            ) {
                                continue;
                            }
                            let Some(name) = ev.name.and_then(|n| n.to_str()) else {
                                continue;
                            };
                            if let Some((dir, files)) = guard.get(&ev.wd)
                                && files.contains(name)
                            {
                                changed_paths.push(dir.join(name));
                            }
                        }
                    }

                    for path in changed_paths {
                        on_shader_changed(state, &path);
                    }
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| anyhow::anyhow!("failed to insert inotify source into calloop: {e:?}"))?;

        Ok(Self {
            watches,
            by_dir: HashMap::new(),
            by_wd,
        })
    }

    pub fn sync_paths<'a>(&mut self, shader_paths: impl Iterator<Item = &'a Path>) {
        let mut desired: HashMap<PathBuf, HashSet<String>> = HashMap::new();

        for path in shader_paths {
            let abs = absolutize_path(path);
            let Some(parent) = abs.parent() else {
                continue;
            };
            let Some(file_name) = abs.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            desired
                .entry(parent.to_path_buf())
                .or_default()
                .insert(file_name.to_string());
        }

        let to_remove: Vec<PathBuf> = self
            .by_dir
            .keys()
            .filter(|dir| !desired.contains_key(*dir))
            .cloned()
            .collect();
        {
            let mut wd_guard = self.by_wd.lock().unwrap();
            for dir in to_remove {
                if let Some(wd) = self.by_dir.remove(&dir) {
                    wd_guard.remove(&wd);
                    let _ = self.watches.remove(wd);
                }
            }

            for (dir, files) in desired {
                if let Some(wd) = self.by_dir.get(&dir) {
                    if let Some(entry) = wd_guard.get_mut(wd) {
                        entry.1 = files;
                    }
                    continue;
                }
                if !dir.exists() {
                    continue;
                }
                match self.watches.add(
                    &dir,
                    WatchMask::CLOSE_WRITE | WatchMask::MOVED_TO | WatchMask::CREATE,
                ) {
                    Ok(wd) => {
                        self.by_dir.insert(dir.clone(), wd.clone());
                        wd_guard.insert(wd, (dir, files));
                    }
                    Err(err) => {
                        tracing::warn!("failed to watch shader dir {}: {err}", dir.display());
                    }
                }
            }
        }
    }
}

pub fn absolutize_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    }
}
