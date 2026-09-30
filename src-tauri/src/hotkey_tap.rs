// hotkey_tap.rs – Doppeltipp auf eine Sondertaste (z. B. zweimal Control)
//
// Ein Tastenkuerzel-Plugin kann keine reinen Modifier-Tasten melden. Statt
// eines systemweiten Tastatur-Hooks (braucht auf macOS die Freigabe
// "Bedienungshilfen", die bei jedem Release neu erteilt werden muesste)
// fragt ein Thread alle 10 ms den Zustand der Sondertasten ab:
//   macOS   CGEventSourceFlagsState + Zaehler der Tastenanschlaege
//   Windows GetAsyncKeyState
//   Linux   XQueryKeymap (nur X11; unter Wayland gibt es keinen Weg)
// Ein Tipp zaehlt nur, wenn waehrenddessen keine andere Taste gedrueckt
// wurde — Strg+C, Strg+C loest also nichts aus.

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// 0 = aus, 1 = Control, 2 = Alt/Option, 3 = Shift, 4 = Command/Windows-Taste
static KEY: AtomicU8 = AtomicU8::new(0);

pub fn set_key(name: &str) {
    KEY.store(match name {
        "control" => 1,
        "alt" => 2,
        "shift" => 3,
        "meta" => 4,
        _ => 0,
    }, Ordering::SeqCst);
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Sample {
    /// Gewaehlte Sondertaste gedrueckt
    pub target: bool,
    /// Eine andere Sondertaste gedrueckt
    pub other_modifier: bool,
    /// Zaehler fuer "irgendeine andere Taste/Maustaste" — aendert er sich, war etwas los.
    pub activity: u64,
}

/// Erkennt zwei kurze Tipps hintereinander. Reine Logik, damit sie ohne
/// Tastatur testbar ist.
#[derive(Debug, Default)]
pub struct TapDetector {
    down_at: Option<Instant>,
    down_activity: u64,
    last_tap: Option<Instant>,
    was_down: bool,
    tainted: bool,
}

/// Laengster Tipp und groesster Abstand zwischen den beiden Tipps.
const MAX_PRESS: Duration = Duration::from_millis(350);
const MAX_GAP: Duration = Duration::from_millis(450);

impl TapDetector {
    /// Neuer Messwert; true = Doppeltipp erkannt.
    pub fn feed(&mut self, s: Sample, now: Instant) -> bool {
        let mut fired = false;
        if s.target && !self.was_down {
            // Taste geht runter
            self.down_at = Some(now);
            self.down_activity = s.activity;
            self.tainted = s.other_modifier;
            if let Some(t) = self.last_tap {
                if now.duration_since(t) > MAX_GAP { self.last_tap = None; }
            }
        } else if s.target && self.was_down {
            if s.other_modifier || s.activity != self.down_activity { self.tainted = true; }
        } else if !s.target && self.was_down {
            // Taste geht hoch
            let clean = !self.tainted && s.activity == self.down_activity
                && self.down_at.map(|d| now.duration_since(d) <= MAX_PRESS).unwrap_or(false);
            if clean {
                match self.last_tap {
                    Some(t) if now.duration_since(t) <= MAX_GAP + MAX_PRESS => {
                        fired = true;
                        self.last_tap = None;
                    }
                    _ => self.last_tap = Some(now),
                }
            } else {
                self.last_tap = None;
            }
            self.down_at = None;
        } else if let Some(t) = self.last_tap {
            // Zwischen den Tipps: eine andere Taste bricht die Folge ab.
            if s.other_modifier || s.activity != self.down_activity || now.duration_since(t) > MAX_GAP {
                self.last_tap = None;
            }
        }
        if !s.target { self.down_activity = s.activity; }
        self.was_down = s.target;
        fired
    }
}

// ============ Plattformen ============

#[cfg(target_os = "macos")]
mod platform {
    use super::Sample;
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceFlagsState(state_id: i32) -> u64;
        fn CGEventSourceCounterForEventType(state_id: i32, event_type: u32) -> u32;
    }
    const COMBINED: i32 = 0; // kCGEventSourceStateCombinedSessionState
    const SHIFT: u64 = 0x0002_0000;
    const CONTROL: u64 = 0x0004_0000;
    const ALT: u64 = 0x0008_0000;
    const CMD: u64 = 0x0010_0000;
    const KEY_DOWN: u32 = 10;
    const LEFT_DOWN: u32 = 1;
    const RIGHT_DOWN: u32 = 3;

    pub struct Poller;
    impl Poller {
        pub fn new() -> Option<Self> { Some(Poller) }
        pub fn sample(&mut self, key: u8) -> Sample {
            let flags = unsafe { CGEventSourceFlagsState(COMBINED) };
            let mask = match key { 1 => CONTROL, 2 => ALT, 3 => SHIFT, _ => CMD };
            let all = SHIFT | CONTROL | ALT | CMD;
            let activity = unsafe {
                CGEventSourceCounterForEventType(COMBINED, KEY_DOWN) as u64
                    + CGEventSourceCounterForEventType(COMBINED, LEFT_DOWN) as u64
                    + CGEventSourceCounterForEventType(COMBINED, RIGHT_DOWN) as u64
            };
            Sample { target: flags & mask != 0, other_modifier: flags & (all & !mask) != 0, activity }
        }
    }
    pub fn supported() -> bool { true }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::Sample;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

    const MODS: [(u8, &[i32]); 4] = [
        (1, &[0x11]),        // VK_CONTROL
        (2, &[0x12]),        // VK_MENU
        (3, &[0x10]),        // VK_SHIFT
        (4, &[0x5B, 0x5C]),  // VK_LWIN, VK_RWIN
    ];

    pub struct Poller { activity: u64, prev_other: bool }
    impl Poller {
        pub fn new() -> Option<Self> { Some(Poller { activity: 0, prev_other: false }) }
        fn down(vk: i32) -> bool { unsafe { (GetAsyncKeyState(vk) as u16 & 0x8000) != 0 } }
        pub fn sample(&mut self, key: u8) -> Sample {
            let mut target = false;
            let mut other_modifier = false;
            for (k, vks) in MODS.iter() {
                let d = vks.iter().any(|v| Self::down(*v));
                if *k == key { target = d } else if d { other_modifier = true }
            }
            // Alle uebrigen Tasten und Maustasten (ohne Modifier und ihre L/R-Varianten)
            let is_mod = |v: i32| matches!(v, 0x10..=0x12 | 0x5B | 0x5C | 0xA0..=0xA5 | 0x14);
            let other = (0x01..=0xFE).filter(|v| !is_mod(*v)).any(Self::down);
            if other && !self.prev_other { self.activity += 1; }
            self.prev_other = other;
            Sample { target, other_modifier, activity: self.activity + other as u64 }
        }
    }
    pub fn supported() -> bool { true }
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use super::Sample;
    use x11_dl::xlib::{Display, Xlib};

    // Tastencodes nach evdev (Standard unter Xorg und XWayland)
    const CONTROL: [u8; 2] = [37, 105];
    const ALT: [u8; 2] = [64, 108];
    const SHIFT: [u8; 2] = [50, 62];
    const META: [u8; 2] = [133, 134];

    pub struct Poller { xlib: Xlib, display: *mut Display, activity: u64, prev_other: bool }
    // Der Poller lebt nur in seinem eigenen Thread.
    unsafe impl Send for Poller {}

    impl Poller {
        pub fn new() -> Option<Self> {
            if !supported() { return None; }
            let xlib = Xlib::open().ok()?;
            let display = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
            if display.is_null() { return None; }
            Some(Poller { xlib, display, activity: 0, prev_other: false })
        }
        pub fn sample(&mut self, key: u8) -> Sample {
            // c_char ist je nach Architektur i8 oder u8
            let mut keys = [0 as std::os::raw::c_char; 32];
            unsafe { (self.xlib.XQueryKeymap)(self.display, keys.as_mut_ptr()); }
            let down = |code: u8| (keys[(code / 8) as usize] as u8) & (1 << (code % 8)) != 0;
            let groups: [(u8, [u8; 2]); 4] = [(1, CONTROL), (2, ALT), (3, SHIFT), (4, META)];
            let mut target = false;
            let mut other_modifier = false;
            for (k, codes) in groups.iter() {
                let d = codes.iter().any(|c| down(*c));
                if *k == key { target = d } else if d { other_modifier = true }
            }
            let mods: Vec<u8> = groups.iter().flat_map(|(_, c)| c.iter().copied()).collect();
            let other = (8u16..=255).map(|c| c as u8).filter(|c| !mods.contains(c)).any(down);
            if other && !self.prev_other { self.activity += 1; }
            self.prev_other = other;
            Sample { target, other_modifier, activity: self.activity + other as u64 }
        }
    }
    impl Drop for Poller {
        fn drop(&mut self) { unsafe { (self.xlib.XCloseDisplay)(self.display); } }
    }
    /// Unter Wayland sieht XQueryKeymap nur Tasten fuer X-Fenster mit Fokus.
    pub fn supported() -> bool {
        std::env::var_os("WAYLAND_DISPLAY").is_none() && std::env::var_os("DISPLAY").is_some()
    }
}

pub fn supported() -> bool { platform::supported() }

/// Startet den Abfrage-Thread. Er schlaeft, solange kein Doppeltipp gewaehlt ist.
pub fn spawn<F: Fn() + Send + 'static>(on_double_tap: F) {
    std::thread::spawn(move || {
        let mut poller: Option<platform::Poller> = None;
        let mut det = TapDetector::default();
        let mut last_key = 0u8;
        loop {
            let key = KEY.load(Ordering::SeqCst);
            if key == 0 {
                poller = None;
                std::thread::sleep(Duration::from_millis(300));
                continue;
            }
            if key != last_key {
                det = TapDetector::default();
                last_key = key;
            }
            if poller.is_none() {
                poller = platform::Poller::new();
                if poller.is_none() {
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
            }
            let s = poller.as_mut().unwrap().sample(key);
            if det.feed(s, Instant::now()) {
                on_double_tap();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(seq: &[(u64, bool, bool, u64)]) -> usize {
        // (ms, target, other_modifier, activity)
        let t0 = Instant::now();
        let mut d = TapDetector::default();
        seq.iter().filter(|(ms, t, o, a)| d.feed(Sample { target: *t, other_modifier: *o, activity: *a }, t0 + Duration::from_millis(*ms))).count()
    }

    #[test]
    fn zwei_kurze_tipps_loesen_aus() {
        assert_eq!(run(&[(0, false, false, 5), (10, true, false, 5), (90, false, false, 5), (200, true, false, 5), (280, false, false, 5)]), 1);
    }

    #[test]
    fn strg_c_zweimal_loest_nicht_aus() {
        // Waehrend Control gedrueckt ist, steigt der Tastenzaehler (C).
        assert_eq!(run(&[(0, false, false, 5), (10, true, false, 5), (40, true, false, 6), (90, false, false, 6),
                         (200, true, false, 6), (230, true, false, 7), (280, false, false, 7)]), 0);
    }

    #[test]
    fn zu_langsam_loest_nicht_aus() {
        assert_eq!(run(&[(0, false, false, 0), (10, true, false, 0), (90, false, false, 0), (900, true, false, 0), (980, false, false, 0)]), 0);
    }

    #[test]
    fn gehaltene_taste_ist_kein_tipp() {
        assert_eq!(run(&[(0, false, false, 0), (10, true, false, 0), (700, false, false, 0), (800, true, false, 0), (850, false, false, 0)]), 0);
    }

    #[test]
    fn andere_sondertaste_bricht_ab() {
        // Control+Shift ist ein Kuerzel, kein Tipp.
        assert_eq!(run(&[(0, false, false, 0), (10, true, true, 0), (80, false, false, 0), (200, true, false, 0), (260, false, false, 0)]), 0);
    }

    #[test]
    fn taste_zwischen_den_tipps_bricht_ab() {
        assert_eq!(run(&[(0, false, false, 0), (10, true, false, 0), (80, false, false, 0), (120, false, false, 1), (200, true, false, 1), (260, false, false, 1)]), 0);
    }

    #[test]
    fn dreifacher_tipp_loest_nur_einmal_aus() {
        assert_eq!(run(&[(0, false, false, 0), (10, true, false, 0), (60, false, false, 0), (150, true, false, 0), (200, false, false, 0),
                         (290, true, false, 0), (340, false, false, 0)]), 1);
    }
}
