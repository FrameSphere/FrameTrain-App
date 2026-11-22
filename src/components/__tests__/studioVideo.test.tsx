// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, screen } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => `asset://${p}`,
}));
vi.mock('../../contexts/LanguageContext', () => ({ useLanguage: () => ({ t: (k: string) => k, language: 'de' }) }));

import StudioVideo from '../studio/StudioVideo';
import type { StudioSample } from '../studio/studioTypes';

const sample = (media: string): StudioSample => ({
  id: `s_${media}`, media, mime: 'video/x-msvideo', status: 'new', ann: { boxes: [] },
  src: { kind: 'import', at: 'x' }, meta: { w: 1, h: 1 }, abs_path: `/m/${media}`,
});
const zeige = (media: string) => render(
  <StudioVideo projectId="p" sample={sample(media)} src={`asset:///m/${media}`} videoRef={() => {}} />);
const video = (c: HTMLElement) => c.querySelector('video') as HTMLVideoElement;

describe('StudioVideo', () => {
  beforeEach(() => { invokeMock.mockReset(); });

  it('spielt ein lesbares Video direkt, ohne Umwandeln', () => {
    const { container } = zeige('a.mp4');
    expect(video(container).getAttribute('src')).toBe('asset:///m/a.mp4');
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it('weicht bei einem Fehler auf die Wiedergabekopie aus', async () => {
    invokeMock.mockResolvedValue('/proxy/b.mp4');
    const { container } = zeige('b.avi');
    fireEvent.error(video(container));
    expect(screen.getByText('studio.media.videoPreparing')).toBeTruthy();
    await waitFor(() => expect(video(container).getAttribute('src')).toBe('asset:///proxy/b.mp4'));
    expect(invokeMock).toHaveBeenCalledWith('studio_playable_video', { projectId: 'p', sampleId: 's_b.avi' });
    // Spielt auch die Kopie nicht, gibt es eine Meldung statt einer Schleife.
    fireEvent.error(video(container));
    expect(screen.getByText('studio.media.videoUnplayable')).toBeTruthy();
    expect(invokeMock).toHaveBeenCalledTimes(1);
  });

  it('zeigt den Fehler des Backends', async () => {
    invokeMock.mockRejectedValue('OpenCV fehlt.');
    const { container } = zeige('c.avi');
    fireEvent.error(video(container));
    await waitFor(() => expect(screen.getByText('OpenCV fehlt.')).toBeTruthy());
  });
});
