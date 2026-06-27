import { useState, useEffect } from 'react';
import {
  X, Check, Copy, ChevronDown, ChevronUp, Plus, Minus,
} from 'lucide-react';
import type { CodeEdit } from '../ai/codeEdits';
import { useLanguage } from '../contexts/LanguageContext';

interface DiffViewerProps {
  edits: CodeEdit[];
  onApply: (editId: string, updatedEdits: CodeEdit[]) => void;
  onApplyAll: (updatedEdits: CodeEdit[]) => void;
  onClose: () => void;
  isApplying?: boolean;
  onEditChange?: (edits: CodeEdit[]) => void;
}

function countLines(text: string): number {
  return text.split('\n').length;
}

function countAddedRemoved(find: string, replace: string): { added: number; removed: number } {
  return {
    removed: countLines(find),
    added: countLines(replace),
  };
}

function DiffLine({ type, content, lineNum }: { type: 'removed' | 'added' | 'context'; content: string; lineNum?: number }) {
  const bgColor = type === 'removed' ? 'bg-red-500/10' : type === 'added' ? 'bg-emerald-500/10' : 'bg-white/[0.02]';
  const borderColor = type === 'removed' ? 'border-l-2 border-red-500/30' : type === 'added' ? 'border-l-2 border-emerald-500/30' : '';
  const textColor = type === 'removed' ? 'text-red-300/70' : type === 'added' ? 'text-emerald-300/70' : 'text-gray-400';
  const prefix = type === 'removed' ? '-' : type === 'added' ? '+' : ' ';

  return (
    <div className={`flex font-mono text-[11px] ${bgColor} ${borderColor} group hover:bg-white/[0.08] transition-colors`}>
      <div className={`w-10 px-2 py-1 text-right select-none ${textColor} flex-shrink-0 bg-white/[0.02]`}>
        {lineNum}
      </div>
      <div className={`w-6 px-1 py-1 text-center flex-shrink-0 font-semibold ${textColor}`}>
        {prefix}
      </div>
      <pre className={`flex-1 py-1 px-2 overflow-x-auto text-gray-200 ${textColor}`}>{content}</pre>
    </div>
  );
}

function EditDiff({ edit, onApply, isApplying }: { edit: CodeEdit; onApply: (updatedEdit: CodeEdit) => void; isApplying?: boolean }) {
  const [expanded, setExpanded] = useState(true);
  const [copied, setCopied] = useState(false);
  const { added, removed } = countAddedRemoved(edit.find, edit.replace);

  const findLines = edit.find.split('\n');
  const replaceLines = edit.replace.split('\n');
  const { t } = useLanguage();

  return (
    <div className="border border-white/10 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-white/[0.04] border-b border-white/10 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-0 hover:bg-white/10 rounded transition-colors"
          >
            {expanded ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
          </button>
          <div className="flex-1 min-w-0">
            <p className="text-xs text-gray-400 font-mono truncate">
              {edit.find.split('\n')[0].slice(0, 60)}...
            </p>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="flex items-center gap-1">
              <Minus className="w-3.5 h-3.5 text-red-400" />
              <span className="text-xs font-medium text-red-300">{removed}</span>
            </div>
            <div className="flex items-center gap-1">
              <Plus className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-xs font-medium text-emerald-300">{added}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Diff Content */}
      {expanded && (
        <div className="bg-slate-900/50 max-h-64 overflow-y-auto">
          {/* Removed Section */}
          <div className="border-b border-white/5">
            <div className="bg-red-500/10 border-b border-red-500/20 px-4 py-2">
              <p className="text-xs font-medium text-red-300 flex items-center gap-2">
                <Minus className="w-3.5 h-3.5" /> {t('diffViewer.removeLabel').replace('{count}', String(findLines.length))}
              </p>
            </div>
            <div>
              {findLines.map((line, i) => (
                <DiffLine key={`del-${i}`} type="removed" content={line} lineNum={i + 1} />
              ))}
            </div>
          </div>

          {/* Added Section */}
          <div>
            <div className="bg-emerald-500/10 border-b border-emerald-500/20 px-4 py-2">
              <p className="text-xs font-medium text-emerald-300 flex items-center gap-2">
                <Plus className="w-3.5 h-3.5" /> {t('diffViewer.addLabel').replace('{count}', String(replaceLines.length))}
              </p>
            </div>
            <div>
              {replaceLines.map((line, i) => (
                <DiffLine key={`add-${i}`} type="added" content={line} lineNum={i + 1} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="bg-white/[0.02] border-t border-white/10 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              navigator.clipboard.writeText(edit.replace);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title={t('diffViewer.copyTooltip')}
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>

        {edit.applied ? (
          <div className="flex items-center gap-2 text-emerald-300 text-xs font-medium">
            <Check className="w-3.5 h-3.5" /> {t('diffViewer.appliedLabel')}
          </div>
        ) : edit.failed ? (
          <div className="flex items-center gap-2 text-red-300 text-xs font-medium">
            <X className="w-3.5 h-3.5" /> {t('diffViewer.failedLabel')}
          </div>
        ) : (
          <button
            onClick={() => onApply({ ...edit, applied: true, failed: false })}
            disabled={isApplying}
            className="px-4 py-2 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-xs font-medium transition-all disabled:opacity-50"
          >
            {t('diffViewer.applyButton')}
          </button>
        )}
      </div>
    </div>
  );
}

export default function DiffViewer({ edits: initialEdits, onApply, onApplyAll, onClose, isApplying, onEditChange }: DiffViewerProps) {
  const [edits, setEdits] = useState(initialEdits);
  const totalRemoved = edits.reduce((acc, e) => acc + countLines(e.find), 0);
  const totalAdded = edits.reduce((acc, e) => acc + countLines(e.replace), 0);
  const allApplied = edits.every(e => e.applied);
  const { t } = useLanguage();

  useEffect(() => {
    setEdits(initialEdits);
  }, [JSON.stringify(initialEdits)]);

  const handleApply = (editId: string, updatedEdit: CodeEdit) => {
    const newEdits = edits.map(e => e.id === editId ? updatedEdit : e);
    setEdits(newEdits);
    onEditChange?.(newEdits);
    onApply(editId, newEdits);
  };

  const handleApplyAll = () => {
    const newEdits = edits.map(e => ({ ...e, applied: e.applied || !e.failed }));
    setEdits(newEdits);
    onEditChange?.(newEdits);
    onApplyAll(newEdits);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-950 rounded-2xl border border-white/10 w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-white/[0.02] flex-shrink-0">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-bold text-white">{t('diffViewer.title')}</h2>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Minus className="w-4 h-4 text-red-400" />
                <span className="text-sm font-medium text-red-300">{t('diffViewer.removedLines').replace('{count}', String(totalRemoved))}</span>
              </div>
              <div className="flex items-center gap-2">
                <Plus className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-emerald-300">{t('diffViewer.addedLines').replace('{count}', String(totalAdded))}</span>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          <p className="text-xs text-gray-500">
            {(edits.length === 1 ? t('diffViewer.changesCount') : t('diffViewer.changesCountPlural')).replace('{count}', String(edits.length))} · {t('diffViewer.appliedCount').replace('{count}', String(edits.filter(e => e.applied).length))}
          </p>
          {edits.map((edit) => (
            <EditDiff
              key={edit.id}
              edit={edit}
              onApply={(updatedEdit) => handleApply(edit.id, updatedEdit)}
              isApplying={isApplying}
            />
          ))}
        </div>

        {/* Footer */}
        <div className="border-t border-white/10 bg-white/[0.02] px-6 py-4 flex items-center justify-between flex-shrink-0">
          <p className="text-xs text-gray-500">
            {allApplied ? (
              <span className="inline-flex items-center gap-1.5">
                <Check className="w-3.5 h-3.5" />
                {t('diffViewer.allApplied')}
              </span>
            ) : (
              t('diffViewer.readyCount').replace('{count}', String(edits.filter(e => !e.applied && !e.failed).length))
            )}
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm font-medium transition-all"
            >
              {t('diffViewer.closeButton')}
            </button>
            {!allApplied && edits.some(e => !e.applied && !e.failed) && (
              <button
                onClick={handleApplyAll}
                disabled={isApplying}
                className="px-4 py-2 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-sm font-medium transition-all disabled:opacity-50"
              >
                {t('diffViewer.applyAllButton')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
