//! Plain-text progress rendering for `cranestudio download` — same
//! print-and-exit treatment as `doctor`/`catalog` ahead of the real
//! ratatui progress bar in M7.

use studio_core::download::Event;

use crate::fmt::bytes;

#[allow(clippy::cast_precision_loss)]
pub fn render_event(event: &Event) {
    match event {
        Event::Started {
            file,
            resume_from,
            total,
        } => {
            if *resume_from > 0 {
                println!(
                    "{file}: resuming from {} of {}",
                    bytes(*resume_from),
                    bytes(*total)
                );
            } else {
                println!("{file}: starting ({})", bytes(*total));
            }
        }
        Event::Progress {
            file,
            downloaded,
            total,
        } => {
            let percent = if *total > 0 {
                (*downloaded as f64 / *total as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "{file}: {} / {} ({percent:.1}%)",
                bytes(*downloaded),
                bytes(*total)
            );
        }
        Event::Verifying { file } => println!("{file}: verifying checksum..."),
        Event::Completed { file } => println!("{file}: done"),
        Event::Cancelled { file } => println!("{file}: cancelled"),
    }
}
