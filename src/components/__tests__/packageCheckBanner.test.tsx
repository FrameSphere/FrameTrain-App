// Hinweis auf fehlende Python-Pakete vor Training/Test.
// Die Gruppen LLM und Generativ sind im Erststart optional — ohne diesen
// Hinweis scheiterte ein Training erst im Python an einem ImportError.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

import PackageCheckBanner from '../PackageCheckBanner';
import { describeMissing } from '../usePackageInstall';

type Cb = (e: { payload: unknown }) => void;
let listeners: Record<string, Cb[]> = {};
const emit = (event: string, payload: unknown) => listeners[event]?.forEach(cb => cb({ payload }));

beforeEach(() => {
  listeners = {};
  mockInvoke.mockReset();
  mockListen.mockReset();
  mockListen.mockImplementation((event: string, cb: Cb) => {
    (listeners[event] ??= []).push(cb);
    return Promise.resolve(() => { listeners[event] = (listeners[event] ?? []).filter(f => f !== cb); });
  });
});

describe('PackageCheckBanner', () => {
  it('zeigt nichts, wenn alles installiert ist', async () => {
    mockInvoke.mockResolvedValue({ group: 'llm', missing: [], python: '/py' });
    const { container } = render(<PackageCheckBanner taskType="causal_lm" />);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('check_task_packages', { taskType: 'causal_lm' }));
    expect(container.textContent).toBe('');
  });

  it('nennt fehlende und zu alte Pakete und installiert die passende Gruppe', async () => {
    let installed = false;
    mockInvoke.mockImplementation((cmd: string) => {
      if (cmd === 'check_task_packages') {
        return Promise.resolve(installed
          ? { group: 'llm', missing: [], python: '/usr/local/bin/python3.12' }
          : { group: 'llm', python: '/usr/local/bin/python3.12', missing: [
              { package: 'peft', installed: false, version: '0.7.1' },
              { package: 'mlx-lm', installed: false, version: null },
            ] });
      }
      return Promise.resolve(null);
    });
    render(<PackageCheckBanner taskType="causal_lm" />);
    expect(await screen.findByText(/peft \(0\.7\.1/)).toBeTruthy();
    expect(screen.getByText(/python3\.12/)).toBeTruthy();

    fireEvent.click(screen.getByRole('button'));
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('install_plugins', { pluginIds: ['llm'] }));
    await waitFor(() => expect(listeners['plugin-install-complete']?.length).toBeGreaterThan(0));
    installed = true;
    act(() => emit('plugin-install-complete', null));
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
  });

  it('zeigt einen Installationsfehler', async () => {
    mockInvoke.mockImplementation((cmd: string) => cmd === 'check_task_packages'
      ? Promise.resolve({ group: 'generative', python: '/py', missing: [{ package: 'diffusers', installed: false }] })
      : Promise.resolve(null));
    render(<PackageCheckBanner taskType="text_to_image_lora" />);
    fireEvent.click(await screen.findByRole('button'));
    await waitFor(() => expect(listeners['plugin-install-progress']?.length).toBeGreaterThan(0));
    act(() => emit('plugin-install-progress', { plugin_id: 'system', status: 'failed', message: 'Kein Internet', progress: 10 }));
    expect(await screen.findByText('Kein Internet')).toBeTruthy();
  });

  it('beschreibt zu alte Pakete mit Version', () => {
    expect(describeMissing([{ package: 'peft', installed: false, version: '0.7.1' }, { package: 'x', installed: false }], 'zu alt'))
      .toBe('peft (0.7.1 zu alt), x');
  });
});
