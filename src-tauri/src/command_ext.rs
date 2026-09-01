// command_ext.rs
// Verhindert unter Windows das Aufblitzen eines Konsolenfensters bei jedem
// Subprozess. std::process::Command oeffnet auf Windows fuer JEDEN Aufruf
// (python, pip, nvidia-smi, taskkill, ...) ein schwarzes cmd-Fenster, solange
// nicht CREATE_NO_WINDOW gesetzt ist. Genau das liess die Installation und den
// laufenden Betrieb "sketchy" wirken: staendig aufpoppende Terminalfenster.
//
// Auf Nicht-Windows-Plattformen ist `.no_window()` ein reiner No-Op, sodass
// derselbe Aufrufcode ueberall unveraendert bleibt.

use std::process::Command;

pub trait NoWindow {
    /// Setzt unter Windows CREATE_NO_WINDOW, damit kein Konsolenfenster
    /// aufblitzt. Ueberall sonst ohne Wirkung.
    fn no_window(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        self.creation_flags(CREATE_NO_WINDOW)
    }
}

#[cfg(not(windows))]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        self
    }
}

/// Erzwingt UTF-8 fuer stdin/stdout/stderr des Python-Subprozesses.
///
/// Auf Windows ist die Standard-Kodierung cp1252 (charmap). Sobald die Engine
/// (oder tqdm/HuggingFace, oder ein User-Dev-Skript) ein Zeichen ausserhalb von
/// cp1252 ausgibt — z.B. ✅ (U+2705) — stirbt der print mit UnicodeEncodeError
/// und das Training endet, bevor es beginnt. PYTHONUTF8=1 aktiviert den UTF-8-
/// Modus des Interpreters von Anfang an; PYTHONIOENCODING ist die Absicherung
/// fuer aeltere Interpreter. Auf Mac/Linux schadet es nicht (dort ist es Default).
pub trait PythonUtf8 {
    fn python_utf8(&mut self) -> &mut Self;
}

impl PythonUtf8 for Command {
    fn python_utf8(&mut self) -> &mut Self {
        self.env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
    }
}
