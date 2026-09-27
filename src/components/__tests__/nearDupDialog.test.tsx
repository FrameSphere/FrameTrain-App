// Beinahe-Dubletten: fehlende Bild-Hashes werden erst gerechnet, dann
// geprueft; je Gruppe bleibt das erste Sample, die anderen sind vorgemerkt.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));

import NearDupDialog from '../studio/NearDupDialog';

const PROJECT = { id: 'sp', name: 'b', modality: 'image', task: 'bbox', target_format: 'yolo_bbox',
  classes: [], created_at: '', updated_at: '' };
const bild = (id: string) => ({ id, media: `ab/${id}.jpg`, mime: 'image/jpeg', status: 'new',
  ann: { boxes: [] }, src: { kind: 'import', at: '' }, meta: { w: 1, h: 1 }, abs_path: `/m/${id}.jpg` });

describe('NearDupDialog', () => {
  beforeEach(() => invokeMock.mockReset());

  it('rechnet fehlende Hashes, zeigt die Gruppe und entfernt die Kopie', async () => {
    let runde = 0;
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_near_duplicates') {
        runde += 1;
        return runde === 1
          ? { kind: 'image', groups: [], checked: 0, missing: [bild('a'), bild('b')] }
          : { kind: 'image', groups: [['a', 'b']], checked: 2, missing: [] };
      }
      if (cmd === 'studio_list_samples') return { total: 2, items: [bild('a'), bild('b')] };
      return null;
    });
    const onDone = vi.fn();
    render(<NearDupDialog project={PROJECT} onClose={vi.fn()} onDone={onDone} hash={async () => 'ffffffffffffffff'} />);

    expect(await screen.findByText('1 Gruppen unter 2 Samples.')).toBeInTheDocument();
    expect(invokeMock).toHaveBeenCalledWith('studio_save_hashes', { projectId: 'sp',
      hashes: { 'ab/a.jpg': 'ffffffffffffffff', 'ab/b.jpg': 'ffffffffffffffff' } });
    const boxen = screen.getAllByRole('checkbox') as HTMLInputElement[];
    expect(boxen.map(b => b.checked)).toEqual([false, true]);   // das erste bleibt

    fireEvent.click(screen.getByRole('button', { name: '1 entfernen' }));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_delete_samples', { projectId: 'sp', sampleIds: ['b'] }));
    expect(onDone).toHaveBeenCalledWith(1);
  });
});
