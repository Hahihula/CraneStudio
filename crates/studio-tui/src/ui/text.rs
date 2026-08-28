//! Word wrapping done by hand, because the chat transcript needs to know how
//! many rows it will occupy *before* it renders: that count is what decides
//! the scroll offset that keeps the newest reply pinned to the bottom.
//! `Paragraph`'s own `Wrap` does the wrapping after that decision is made, so
//! it can't answer the question.

/// Wraps `text` to `width` columns, breaking on spaces where possible and
/// mid-word only when a single word is longer than the whole line. Existing
/// newlines are honored. Never returns an empty vector — a blank input is one
/// blank line, so a message with a trailing newline keeps its shape.
#[must_use]
pub fn wrap(text: &str, width: u16) -> Vec<String> {
    let width = width.max(1) as usize;
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        let emitted_before = out.len();
        let mut line = String::new();
        for word in paragraph.split(' ') {
            if word.chars().count() > width {
                if !line.is_empty() {
                    out.push(std::mem::take(&mut line));
                }
                for chunk in chunks(word, width) {
                    out.push(chunk);
                }
                continue;
            }
            let projected = if line.is_empty() {
                word.chars().count()
            } else {
                line.chars().count() + 1 + word.chars().count()
            };
            if projected > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        // A paragraph always contributes at least one row, so a blank line in
        // the source stays a blank line on screen — but a paragraph that ended
        // exactly on a wrap boundary doesn't gain a spurious empty one.
        if !line.is_empty() || out.len() == emitted_before {
            out.push(line);
        }
    }
    out
}

fn chunks(word: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in word.chars() {
        if current.chars().count() == width {
            out.push(std::mem::take(&mut current));
        }
        current.push(ch);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Shortens `text` to `width` columns with a trailing ellipsis — for list
/// rows, where a wrapped path would push every following row out of place.
#[must_use]
pub fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".to_string();
    }
    let kept: String = text.chars().take(width - 1).collect();
    format!("{kept}…")
}

/// Keeps the *end* of `text` instead of the start — the right choice for file
/// paths, where the filename matters more than the leading directories.
#[must_use]
pub fn truncate_start(text: &str, width: usize) -> String {
    let count = text.chars().count();
    if count <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".to_string();
    }
    let skipped = count - (width - 1);
    let kept: String = text.chars().skip(skipped).collect();
    format!("…{kept}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_on_spaces() {
        assert_eq!(wrap("one two three", 7), vec!["one two", "three"]);
    }

    #[test]
    fn keeps_existing_newlines() {
        assert_eq!(wrap("a\nb", 10), vec!["a", "b"]);
    }

    #[test]
    fn breaks_a_word_longer_than_the_line() {
        assert_eq!(wrap("aaaaaa", 3), vec!["aaa", "aaa"]);
    }

    #[test]
    fn a_blank_line_survives() {
        assert_eq!(wrap("", 10), vec![""]);
        assert_eq!(wrap("a\n\nb", 10), vec!["a", "", "b"]);
    }

    #[test]
    fn truncation_marks_what_it_dropped() {
        assert_eq!(truncate("abcdef", 4), "abc…");
        assert_eq!(truncate("abc", 4), "abc");
        assert_eq!(truncate_start("/a/b/model.gguf", 8), "…el.gguf");
    }
}
