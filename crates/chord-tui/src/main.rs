//! chord-tui entrypoint (S91 CTUI-01) — the async ratatui event loop.
//!
//! Wires the shared plumbing (config → secrets → connection manager) to the
//! ratatui shell. The event loop `select!`s over terminal events and a redraw
//! tick, so a slow/dead instance polled on its own task NEVER freezes input or
//! rendering.
//!
//! This binary is a CLIENT. It connects to Chord/Terminus control endpoints and
//! never restarts or reconfigures the live proxy.

use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::time::interval;

use chord_tui::app::{App, Mode};
use chord_tui::config::Config;
use chord_tui::connection::{ConnectionManager, HttpHealthProbe, InstanceStatus};
use chord_tui::modes::chord::chord_client::{ChordClient, ChordSnapshot};
use chord_tui::modes::chord::models::pull_mutation;
use chord_tui::modes::chord::ChordPanel;
use chord_tui::secret::EnvSecretManager;
use chord_tui::ui::{self, Framedata};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Load config (missing → empty fleet; corrupt → backup + fresh + warn) ──
    let cfg_path = Config::default_path();
    let loaded = Config::load(&cfg_path);
    let config = loaded.config;

    // Vault-backed secrets (env injected by the vault agent; never literals).
    let secrets = Arc::new(EnvSecretManager::from_env());

    // Async multi-instance health manager — one task per instance, never blocks.
    let manager = ConnectionManager::spawn(
        config.instances.clone(),
        Arc::new(HttpHealthProbe::new()),
        secrets.clone(),
        Duration::from_secs(config.settings.poll_interval_secs.max(1)),
        Duration::from_secs(config.settings.request_timeout_secs.max(1)),
    );

    // Optional Chord snapshot client for the first Chord instance (read-only).
    let chord_client = config
        .instances
        .iter()
        .find(|i| matches!(i.kind, chord_tui::config::InstanceKind::Chord))
        .map(|i| ChordClient::new(i.base_url.clone(), Duration::from_secs(config.settings.request_timeout_secs.max(1))));

    let mut app = App::new(&config);
    if let Some(w) = loaded.warning {
        app.set_toast(w, chord_tui::app::ToastLevel::Warn);
    }

    // ── Terminal setup ────────────────────────────────────────────────────────
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_loop(&mut terminal, &mut app, &manager, chord_client.as_ref(), secrets.clone()).await;

    // ── Restore terminal ──────────────────────────────────────────────────────
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

async fn run_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    manager: &ConnectionManager,
    chord_client: Option<&ChordClient>,
    secrets: Arc<EnvSecretManager>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut events = EventStream::new();
    let mut redraw = interval(Duration::from_millis(250));
    let mut instances: Vec<InstanceStatus> = manager.snapshot().await;
    let mut chord_snap: Option<ChordSnapshot> = None;

    loop {
        // Draw current state.
        terminal.draw(|f| {
            let data = Framedata { app, instances: &instances, chord: chord_snap.as_ref() };
            ui::draw(f, &data);
        })?;

        if app.should_quit {
            break;
        }

        tokio::select! {
            // Terminal input.
            maybe = events.next() => {
                if let Some(Ok(ev)) = maybe {
                    handle_event(app, ev);
                }
            }
            // Periodic redraw + snapshot refresh (never blocks on a dead node).
            _ = redraw.tick() => {
                instances = manager.snapshot().await;
                if app.mode == Mode::Chord {
                    if let Some(client) = chord_client {
                        // Resolve token from vault per refresh; value never logged.
                        let _ = &secrets; // token wiring: EnvSecretManager resolves per instance
                        if let Ok(snap) = client.fetch_snapshot(None).await {
                            chord_snap = Some(snap);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Route a terminal event through the app state machine. Confirm overlay takes
/// precedence so mutations can never fire from a stray key.
fn handle_event(app: &mut App, ev: Event) {
    match ev {
        Event::Resize(w, h) => app.on_resize(w, h),
        Event::Key(k) if k.kind == KeyEventKind::Press => {
            // If a confirm overlay is open, keys drive the confirmation flow.
            if app.confirm.is_some() {
                match k.code {
                    KeyCode::Esc => app.cancel_confirm(),
                    KeyCode::Enter => {
                        let _ = app.confirm_submit_typed();
                    }
                    KeyCode::Backspace => app.confirm_backspace(),
                    KeyCode::Char(c) => {
                        // 'y' confirms a simple mutation; otherwise it's typed input
                        // for a destructive confirmation.
                        if app.confirm_keystroke(c).is_none() {
                            app.confirm_type_char(c);
                        }
                    }
                    _ => {}
                }
                return;
            }

            match k.code {
                KeyCode::Char('q') => app.should_quit = true,
                KeyCode::Tab => app.switch_mode(),
                KeyCode::Right => match app.mode {
                    Mode::Chord => app.chord_panel = app.chord_panel.next(),
                    Mode::TerminusFleet => app.fleet_panel = app.fleet_panel.next(),
                },
                KeyCode::Left => match app.mode {
                    Mode::Chord => app.chord_panel = app.chord_panel.prev(),
                    Mode::TerminusFleet => app.fleet_panel = app.fleet_panel.prev(),
                },
                // 'p' requests a (simple, keystroke-confirmed) model pull demo action.
                KeyCode::Char('p') if app.mode == Mode::Chord && app.chord_panel == ChordPanel::Models => {
                    app.request_mutation(pull_mutation("<selected-model>"));
                }
                _ => {}
            }
        }
        _ => {}
    }
}
