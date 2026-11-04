// KI-Verbindungen (MCP, Strang B): Einstellungs-Tab und Hilfsfunktionen.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => `asset://${p}`,
}));

import AIConnectionsSettings from '../settings/AIConnectionsSettings';
import {
  claudeCodeCommand, cursorJson, connectErrorKey, groupApprovals, parseRemoteStatus, loadApprovalsApi,
  type ApprovalsApi, type McpStatus,
} from '../settings/mcpConnections';
import { LanguageProvider } from '../../contexts/LanguageContext';

const TOKEN = 'ftmcp_0123456789abcdef0123456789abcdef';

function status(over: Partial<McpStatus> = {}, cfg: Partial<McpStatus['config']> = {}): McpStatus {
  return {
    config: { enabled: true, port: 47861, token: TOKEN, deviceId: 'd', remoteEnabled: false, ...cfg },
    server: { running: true, url: 'http://127.0.0.1:47861/mcp', error: null },
    relay: { running: false, connected: false, lastHeartbeat: null, lastError: null, jobsHandled: 0 },
    manifestTools: 1, manifestLoaded: true, pendingCalls: 0,
    recentCalls: [], localClients: [],
    exePath: '/Applications/FrameTrain.app/Contents/MacOS/FrameTrain', exeTranslocated: false,
    claudeDesktop: { configPath: '/x/claude_desktop_config.json', installed: true, configured: false },
    ...over,
  };
}

function wrap(ui: React.ReactElement) {
  localStorage.setItem('ft_language', 'de');
  return render(<LanguageProvider>{ui}</LanguageProvider>);
}

describe('mcpConnections Hilfen', () => {
  it('Claude-Code-Befehl und Cursor-JSON enthalten URL und Token', () => {
    expect(claudeCodeCommand(47861, TOKEN)).toBe(
      `claude mcp add --transport http frametrain http://127.0.0.1:47861/mcp --header "Authorization: Bearer ${TOKEN}"`,
    );
    const c = JSON.parse(cursorJson(5000, TOKEN));
    expect(c.mcpServers.frametrain.url).toBe('http://127.0.0.1:5000/mcp');
    expect(c.mcpServers.frametrain.headers.Authorization).toBe(`Bearer ${TOKEN}`);
  });

  it('Fehlercodes, Remote-Status und Freigaben-Gruppen', () => {
    expect(connectErrorKey('CLAUDE_NOT_FOUND')).toBe('mcp.settings.claudeDesktop.notFound');
    expect(connectErrorKey('CLAUDE_CONFIG_INVALID: x')).toBe('mcp.settings.claudeDesktop.invalidConfig');
    expect(connectErrorKey('anders')).toBeNull();
    expect(parseRemoteStatus(null)).toEqual({ connections: [], recentJobs: [] });
    expect(parseRemoteStatus({ connections: [{ clientName: 'ChatGPT' }] }).connections).toHaveLength(1);
    const g = groupApprovals([
      { client: 'Cursor', tool: 'training_start' },
      { client: 'Claude', tool: 'models_delete' },
      { client: 'Claude', tool: 'datasets_delete' },
      { client: '', tool: 'x' },
    ]);
    expect(g.map(x => x.client)).toEqual(['Claude', 'Cursor']);
    expect(g[0].items.map(i => i.tool)).toEqual(['datasets_delete', 'models_delete']);
  });

  it('optionales approvals-Modul fehlt nicht hart', async () => {
    const api = await loadApprovalsApi();
    // Entweder Strang A ist da (Funktionen vorhanden) oder es kommt null – nie ein Fehler.
    if (api) expect(typeof api.listApprovals).toBe('function');
    else expect(api).toBeNull();
  });
});

describe('Einstellungs-Tab KI-Verbindungen', { timeout: 15000 }, () => {
  beforeEach(() => invokeMock.mockReset());

  it('zeigt Status, versteckt den Token und speichert Schalter', async () => {
    let saved: Record<string, unknown> | null = null;
    invokeMock.mockImplementation((cmd: string, args: Record<string, unknown>) => {
      if (cmd === 'mcp_get_status') return Promise.resolve(status());
      if (cmd === 'mcp_save_settings') { saved = args; return Promise.resolve(status({}, { enabled: args.enabled as boolean, remoteEnabled: args.remoteEnabled as boolean })); }
      if (cmd === 'mcp_remote_status') return Promise.resolve({ connections: [{ id: 'c1', clientName: 'ChatGPT', createdAt: '2026-11-04T10:00:00Z', lastUsedAt: null }], recentJobs: [] });
      return Promise.resolve(null);
    });
    wrap(<AIConnectionsSettings approvalsApi={null} />);
    expect(await screen.findByText('Läuft auf http://127.0.0.1:47861/mcp')).toBeInTheDocument();
    // Token nirgends im Klartext sichtbar
    expect(document.body.textContent).not.toContain(TOKEN);
    expect(screen.getByText(/claude mcp add --transport http frametrain/)).toBeInTheDocument();

    // Fernzugriff einschalten -> speichert und laedt die verbundenen Clients
    const switches = screen.getAllByRole('switch');
    fireEvent.click(switches[1]);
    await waitFor(() => expect(saved).toEqual({ enabled: true, port: 47861, remoteEnabled: true }));
    expect(await screen.findByText('ChatGPT')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Trennen'));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('mcp_remote_disconnect', { connectionId: 'c1' }));

    // Token zeigen
    fireEvent.click(screen.getByLabelText('Token zeigen'));
    expect(screen.getAllByText(TOKEN).length).toBeGreaterThan(0);
  });

  it('Mit Claude Desktop verbinden zeigt Sicherung und Haken', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'mcp_get_status') return Promise.resolve(status({}, { enabled: false }));
      if (cmd === 'mcp_connect_claude_desktop') return Promise.resolve({
        configPath: '/x/claude_desktop_config.json', backupPath: '/x/backup.json',
        status: status({ claudeDesktop: { configPath: '/x', installed: true, configured: true } }),
      });
      return Promise.resolve(null);
    });
    wrap(<AIConnectionsSettings approvalsApi={null} />);
    fireEvent.click(await screen.findByText('Mit Claude Desktop verbinden'));
    expect(await screen.findByText(/Sicherung unter \/x\/backup\.json/)).toBeInTheDocument();
    expect(screen.getByText('Erneut eintragen')).toBeInTheDocument();
  });

  it('Claude Desktop fehlt: Knopf aus, Hinweis da; Fehlercode wird uebersetzt', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'mcp_get_status') return Promise.resolve(status({ claudeDesktop: { configPath: null, installed: false, configured: false } }));
      return Promise.resolve(null);
    });
    wrap(<AIConnectionsSettings approvalsApi={null} />);
    const btn = await screen.findByText('Mit Claude Desktop verbinden');
    expect(btn.closest('button')).toBeDisabled();
    expect(screen.getByText(/wurde auf diesem Rechner nicht gefunden/)).toBeInTheDocument();
  });

  it('Token neu erzeugen fragt erst nach (Modal im body)', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'mcp_get_status') return Promise.resolve(status());
      if (cmd === 'mcp_rotate_token') return Promise.resolve(status({}, { token: 'ftmcp_ffffffffffffffffffffffffffffffff' }));
      return Promise.resolve(null);
    });
    wrap(<AIConnectionsSettings approvalsApi={null} />);
    fireEvent.click(await screen.findByLabelText('Neuen Token erzeugen'));
    const dialog = await screen.findByRole('dialog');
    expect(dialog.parentElement?.parentElement).toBe(document.body);
    expect(invokeMock).not.toHaveBeenCalledWith('mcp_rotate_token');
    fireEvent.click(within(dialog).getByText('Neuen Token erzeugen'));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('mcp_rotate_token'));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('Freigaben je Client listen und widerrufen', async () => {
    invokeMock.mockImplementation((cmd: string) => cmd === 'mcp_get_status' ? Promise.resolve(status({
      recentCalls: [{ tool: 'models_list', clientName: 'Claude Desktop', source: 'local', isError: false, at: '2026-11-04T10:00:00Z', durationMs: 5 }],
    })) : Promise.resolve(null));
    let list = [{ client: 'Claude Desktop', tool: 'training_start' }, { client: 'Cursor', tool: 'models_delete' }];
    const api: ApprovalsApi = {
      listApprovals: () => list,
      revokeApproval: vi.fn((client: string, tool: string) => { list = list.filter(a => !(a.client === client && a.tool === tool)); }),
    };
    wrap(<AIConnectionsSettings approvalsApi={api} />);
    expect(await screen.findByText('training_start')).toBeInTheDocument();
    expect(screen.getByText('models_list')).toBeInTheDocument(); // letzte Aufrufe
    fireEvent.click(screen.getByLabelText('training_start für Claude Desktop widerrufen'));
    expect(api.revokeApproval).toHaveBeenCalledWith('Claude Desktop', 'training_start');
    await waitFor(() => expect(screen.queryByText('training_start')).toBeNull());
    expect(screen.getByText('models_delete')).toBeInTheDocument();
  });
});
