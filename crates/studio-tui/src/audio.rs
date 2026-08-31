//! In-process WAV playback for the TTS Playground, on a dedicated thread that
//! owns the `!Send` `rodio` stream. With no output device, playback is a no-op.

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

enum Cmd {
    Play(PathBuf),
    Stop,
}

pub struct Player {
    tx: Sender<Cmd>,
    available: bool,
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

impl Player {
    /// Spawns the playback thread; reports whether an output device opened.
    #[must_use]
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("tts-playback".into())
            .spawn(move || run(&rx, &ready_tx))
            .ok();
        let available = ready_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap_or(false);
        Player { tx, available }
    }

    /// Plays `path`, replacing any current playback. Returns `false` with no
    /// audio device.
    #[must_use]
    pub fn play(&self, path: &Path) -> bool {
        self.available && self.tx.send(Cmd::Play(path.to_path_buf())).is_ok()
    }

    pub fn stop(&self) {
        let _ = self.tx.send(Cmd::Stop);
    }
}

fn run(rx: &Receiver<Cmd>, ready: &Sender<bool>) {
    let Ok((_stream, handle)) = rodio::OutputStream::try_default() else {
        let _ = ready.send(false);
        return;
    };
    let Ok(sink) = rodio::Sink::try_new(&handle) else {
        let _ = ready.send(false);
        return;
    };
    let _ = ready.send(true);
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Play(path) => {
                sink.stop();
                if let Ok(file) = std::fs::File::open(&path)
                    && let Ok(source) = rodio::Decoder::new(BufReader::new(file))
                {
                    sink.append(source);
                    sink.play();
                }
            }
            Cmd::Stop => sink.stop(),
        }
    }
}
