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
