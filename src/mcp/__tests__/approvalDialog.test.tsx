// Freigabe-Dialog: zeigt Client, Werkzeug und Argumente; Knoepfe beantworten
// die Anfrage; bei Loeschen gibt es kein "immer erlauben".

import { afterEach, describe, expect, it } from 'vitest';
import { act, render, screen, fireEvent, cleanup } from '@testing-library/react';
import ApprovalDialog, { formatArgs } from '../ApprovalDialog';
import { requestApprovalViaDialog, resetApprovalQueue } from '../approval';
import { defineTool, ok } from '../util';

const start = defineTool({ name: 'training_start', title: 'Training starten', description: 'x'.repeat(50), approval: 'always', run: async () => ok('') });
const del = defineTool({ name: 'models_delete', title: 'Modell loeschen', description: 'x'.repeat(50), approval: 'always', annotations: { destructiveHint: true }, run: async () => ok('') });

afterEach(() => { cleanup(); resetApprovalQueue(); localStorage.clear(); });

describe('ApprovalDialog', () => {
  it('ist ohne Anfrage unsichtbar', () => {
    render(<ApprovalDialog />);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('zeigt die Anfrage und liefert "immer erlauben"', async () => {
    render(<ApprovalDialog />);
    let p!: Promise<unknown>;
    act(() => { p = requestApprovalViaDialog({ tool: start, args: { model_id: 'm1', config: { epochs: 3 } }, client: { name: 'Claude Desktop' }, source: 'local' }); });
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(screen.getByText(/Claude Desktop möchte/)).toBeTruthy();
    expect(screen.getByText('Training starten')).toBeTruthy();
    expect(screen.getByText('m1')).toBeTruthy();
    act(() => { fireEvent.click(screen.getByText('Für Claude Desktop immer erlauben')); });
    await expect(p).resolves.toBe('always');
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('Loeschen: Warnung, kein "immer erlauben", Escape lehnt ab', async () => {
    render(<ApprovalDialog />);
    let p!: Promise<unknown>;
    act(() => { p = requestApprovalViaDialog({ tool: del, args: { model_id: 'm1' }, client: { name: 'ChatGPT' }, source: 'remote' }); });
    expect(screen.getByText(/nicht rückgängig/)).toBeTruthy();
    expect(screen.getByText(/Fernzugriff/)).toBeTruthy();
    expect(screen.queryByText(/immer erlauben/)).toBeNull();
    act(() => { fireEvent.keyDown(window, { key: 'Escape' }); });
    await expect(p).resolves.toBe('deny');
  });

  it('kuerzt lange Argumente und Bilder', () => {
    const rows = formatArgs({ script: 'x'.repeat(1000), image_base64: 'A'.repeat(4096), n: 3 });
    expect(rows[0].value.length).toBeLessThan(450);
    expect(rows[1].value).toBe('(3 KB)');
    expect(rows[2].value).toBe('3');
  });
});
