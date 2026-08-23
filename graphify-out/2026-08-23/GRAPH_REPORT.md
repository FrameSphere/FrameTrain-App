# Graph Report - desktop-app  (2026-08-19)

## Corpus Check
- 228 files · ~340,961 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2635 nodes · 5787 edges · 150 communities (135 shown, 15 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 85 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `dd8198ad`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- dataset_manager.rs
- training_manager.rs
- TrainingPanel.tsx
- Database
- ir.py
- DevTestPanel.tsx
- SynapseAICoachPanel.tsx
- Dashboard.tsx
- .status
- Manager (Cloudflare Worker + D1)
- AnalysisPanel.tsx
- laboratory_manager.rs
- tauri.conf.json
- plugin_commands.rs
- test_manager.rs
- String
- supported_architectures
- TrainingDashboard.tsx
- train_engine.py
- model_manager.rs
- LaboratoryPanel.tsx
- load_canvas_model
- useLanguage
- DevTrainPanel.tsx
- plugins/datasetCompat.ts
- MessageProtocol
- SynapseBuilder.tsx
- CanvasPlugin
- compilerOptions
- FloatingAICoach.tsx
- ModelLibrary.tsx
- yolo/manifest.json
- Settings.tsx
- usePageContext
- canvas/manifest.json
- image_classification/manifest.json
- .status
- OpenLibraryModal.tsx
- nodeTypes.ts
- SynapseAIPanel.tsx
- analysis_manager.rs
- ImageClassificationPlugin
- dev_trainer.rs
- supported_architectures
- devDependencies
- dependencies
- TrainingConsole.tsx
- Plugin
- ui.test.tsx
- synapseAgent.ts
- codeGenerator.ts
- graph-shape-validation.ts
- definitions
- definitions
- supported_architectures
- DatasetUpload.tsx
- image-classification/index.ts
- types.ts
- properties
- .error
- PowerState
- yolo/index.ts
- main.rs
- properties
- permissions
- DynamicGraphModule
- permissions
- supported_architectures
- hf-encoder/detect.ts
- webviews
- webviews
- generate-icons.py
- auth.rs
- compilerOptions
- codeEdits.ts
- synapseAgentTools.ts
- ModelConfig
- CapabilityRemote
- CapabilityRemote
- ModelServer
- test_engine/plugins/seq_classification/manifest.json
- errorReport.ts
- AIKnowledgeBaseSmart.ts
- permissions
- api_config.rs
- package.json
- scripts
- Capability
- Capability
- desktop-schema.json
- macOS-schema.json
- ModelManager.tsx
- synapseOps.ts
- local
- local
- run_dataset_classification
- ShellScopeEntryAllowedArgs
- Target
- postcss
- @testing-library/react
- @types/node
- @types/react
- @types/react-dom
- vitest
- generate-placeholder.sh
- canvas/__init__.py
- vite-env.d.ts
- frametrain-desktop
- App Icon (128x128@2x, FrameTrain)
- FrameTrain App Icon (128x128)
- FrameTrain App Icon (32x32)
- FrameTrain Favicon (App Icon)
- Plugin
- NotificationContext.tsx
- DiffViewer.tsx
- TestConfig
- Plugin
- Plugin
- Plugin
- TestProtocol
- test_engine/plugins/audio_classification/manifest.json
- test_engine/plugins/hf_image_classification/manifest.json
- test_engine/plugins/seq2seq/manifest.json
- DatasetInfo
- Identifier

## God Nodes (most connected - your core abstractions)
1. `useLanguage()` - 123 edges
2. `AppState` - 63 edges
3. `Database` - 38 edges
4. `useTheme()` - 36 edges
5. `SynapseBuilderInner()` - 32 edges
6. `useNotification()` - 31 edges
7. `usePageContext()` - 25 edges
8. `MessageProtocol` - 24 edges
9. `split_dataset_in_half()` - 23 edges
10. `useAISettings()` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Auto-Fix Pipeline (App-Fehler → Manager → Triage-Agent → Review → Merge)` --semantically_similar_to--> `SynapseAICoach`  [INFERRED] [semantically similar]
  .claude/AUTOMATION_SETUP.md → src/components/synapse/ai/SynapseAICoach.ts
- `build-tauri job (matrix: macOS aarch64, Ubuntu, Windows)` --references--> `index.html – Vite/Tauri Entry Point`  [INFERRED]
  .github/workflows/release.yml → index.html
- `Synapse AI Coach Integration Guide` --references--> `SynapseBuilder()`  [EXTRACTED]
  src/components/synapse/ai/INTEGRATION_GUIDE.md → src/components/synapse/SynapseBuilder.tsx
- `SynapseBuilder()` --shares_data_with--> `currentModelConfig state`  [EXTRACTED]
  src/components/synapse/SynapseBuilder.tsx → src/components/synapse/ai/INTEGRATION_GUIDE.md
- `Auto-Triage FrameTrain Errors GitHub Workflow` --references--> `ANTHROPIC_API_KEY (GitHub Actions secret)`  [EXTRACTED]
  .github/workflows/auto-triage.yml → .claude/AUTOMATION_SETUP.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Release workflow job dependency chain (create-release → build-tauri → generate-latest-json → notify-release)** — github_workflows_release_create_release_job, github_workflows_release_build_tauri_job, github_workflows_release_generate_latest_json_job, github_workflows_release_notify_release_job [EXTRACTED 1.00]
- **Synapse AI Coach system components (core logic, sub-modules, and UI panel)** — src_components_synapse_ai_synapseaicoach_synapseaicoach, src_components_synapse_ai_synapseaicoach_errorparser, src_components_synapse_ai_synapseaicoach_graphanalyzer, src_components_synapse_ai_synapseaicoach_graphautofixer, src_components_synapse_ai_synapseaicoachpanel_synapseaicoachpanel [EXTRACTED 1.00]
- **Auto-Fix Pipeline: error capture, triage agent, and Manager review loop** — claude_automation_setup_auto_fix_pipeline, claude_automation_setup_manager, claude_commands_triage_errors_triage_agent, github_workflows_auto_triage_job [INFERRED 0.85]

## Communities (150 total, 15 thin omitted)

### Community 0 - "dataset_manager.rs"
Cohesion: 0.09
Nodes (107): Client, add_files_to_dataset(), alte_ascii_namen_werden_erkannt(), analyze_dataset_path(), check_basename_pairing(), collect_extensions(), collect_files(), collect_files_recursive() (+99 more)

### Community 1 - "training_manager.rs"
Cohesion: 0.06
Nodes (74): CanvasInferenceResult, CanvasModelInfo, CanvasNetworkResult, check_training_requirements(), copy_dir(), create_canvas_network_model(), create_version(), default_lora_mods() (+66 more)

### Community 2 - "TrainingPanel.tsx"
Cohesion: 0.09
Nodes (30): applyCoachConfig(), CoachCommand, CoachConfigPatch, consumePendingCoachCommand(), consumePendingCoachConfig(), getRecommendedParams(), onApplyCoachConfig(), onCoachCommand() (+22 more)

### Community 3 - "Database"
Cohesion: 0.08
Nodes (36): Connection, Database, Dataset, get_data_directory(), get_database_path(), Model, ModelVersion, Option (+28 more)

### Community 4 - "ir.py"
Cohesion: 0.10
Nodes (26): get_dataloaders(), _infer_input_features(), CanvasGraphIR, DataLoader, DataLoaders for Canvas IR training., CanvasGraphIR, IRDataSpec, IREdge (+18 more)

### Community 5 - "DevTestPanel.tsx"
Cohesion: 0.12
Nodes (29): AutoMode, buildAutoSystemPrompt(), parseAutoAction(), extractFullPythonCode(), AppliedEditInfo, calculateAffectedLines(), ChatSession, CodeAISidebar() (+21 more)

### Community 6 - "SynapseAICoachPanel.tsx"
Cohesion: 0.06
Nodes (39): adjustParamsFix(), applyAutoFix(), AutoFixResult, insertBridgeFix(), TODO: Implementiere topologisches Sorting der Nodes, TODO: Toast anzeigen, TODO: Error Toast anzeigen, removeNodeFix() (+31 more)

### Community 7 - "Dashboard.tsx"
Cohesion: 0.09
Nodes (25): ApiKeyValidation, CloseDialogProps, NOTE: synapse_sessions_${userId} NICHT löschen - Sessions sind User-Daten und…, UserData, Dashboard(), DashboardProps, UserData, View (+17 more)

### Community 8 - ".status"
Cohesion: 0.21
Nodes (10): handle_exception(), is_oom(), is_torch_ecosystem_conflict(), load_plugin(), Exception, TestConfig, FrameTrain – Test Engine ========================= Plugin-basierter…, torchvision/torchaudio passen nicht zur installierten torch-Version (z. B.… (+2 more)

### Community 9 - "Manager (Cloudflare Worker + D1)"
Cohesion: 0.05
Nodes (46): ANTHROPIC_API_KEY (GitHub Actions secret), app_errors D1 table (extended: error_group, triage_status, occurrences, screen), Auto-Fix Pipeline (App-Fehler → Manager → Triage-Agent → Review → Merge), AUTOMATION_SECRET (Cloudflare secret), Nightly cron routine (0 1 * * * ≈ 03:00 Berlin, or manual workflow_dispatch), FrameTrain Auto-Fix Pipeline Setup Guide, src/utils/errorReport.ts – installGlobalErrorReporting(), fix_proposals D1 table (+38 more)

### Community 10 - "AnalysisPanel.tsx"
Cohesion: 0.08
Nodes (39): setRecommendedParams(), AnalysisPanel(), buildFullContext(), AnalysisPanelProps, asFiniteNumber(), BigLossChart(), BOOLEAN_PARAM_KEYS, buildAnalysisSystemPrompt() (+31 more)

### Community 11 - "laboratory_manager.rs"
Cohesion: 0.15
Nodes (38): BufWriter, ChildStdin, Receiver, get_canvas_server_path(), get_model_server_path(), get_python_path(), get_version_info(), get_version_path() (+30 more)

### Community 12 - "tauri.conf.json"
Cohesion: 0.05
Nodes (39): **, https://github.com/FrameSphere/FrameTrain-App/releases/latest/download/latest.json, icons/128x128@2x.png, icons/128x128.png, icons/32x32.png, icons/icon.icns, icons/icon.ico, python (+31 more)

### Community 13 - "plugin_commands.rs"
Cohesion: 0.16
Nodes (37): GpuInfo, check_dependency_status(), check_first_launch(), check_package_installed(), DependencyStatus, detect_gpu(), find_valid_python(), get_available_plugins() (+29 more)

### Community 14 - "test_manager.rs"
Cohesion: 0.20
Nodes (37): default_mode(), default_task_type(), export_hard_examples(), get_active_test_job(), get_current_test(), get_models_dir(), get_python_path(), get_test_engine_path() (+29 more)

### Community 15 - "String"
Cohesion: 0.16
Nodes (27): ModelInfo, calculate_directory_size(), copy_dir_recursive_export(), Database, delete_model_version(), export_model_version(), get_version_path_for_ui(), list_model_versions() (+19 more)

### Community 16 - "supported_architectures"
Cohesion: 0.06
Nodes (35): albert, bert, camembert, deberta, deberta-v2, distilbert, electra, ernie (+27 more)

### Community 17 - "TrainingDashboard.tsx"
Cohesion: 0.17
Nodes (18): firstUsableLoss(), lossImprovementPct(), LossPointLike, BigLossChart(), ConfigSummary(), ErrorCategory, EVENT_COLORS, EVENT_ICONS (+10 more)

### Community 18 - "train_engine.py"
Cohesion: 0.07
Nodes (17): Any, handle_exception(), is_oom(), is_torch_ecosystem_conflict(), load_plugin(), LoggingTee, main(), Orchestrator (+9 more)

### Community 19 - "model_manager.rs"
Cohesion: 0.18
Nodes (37): calc_speed_and_eta(), calculate_dir_size(), calculate_dir_size_inner(), cleanup_incomplete_download(), copy_dir_recursive(), delete_model(), detect_model_type(), download_huggingface_model() (+29 more)

### Community 20 - "LaboratoryPanel.tsx"
Cohesion: 0.10
Nodes (31): AnalysisView(), deleteSession(), extractLabelField(), extractTextField(), getDisplayText(), getSideInfo(), LABEL_KEYS, LabInputKind (+23 more)

### Community 21 - "load_canvas_model"
Cohesion: 0.11
Nodes (21): CanvasInferenceServer, emit(), emit_error(), main(), _find_model_pt(), load_canvas_model(), _load_ir_from_checkpoint(), _load_ir_from_metadata() (+13 more)

### Community 22 - "useLanguage"
Cohesion: 0.07
Nodes (39): DatasetFileManager(), FileInfo, formatBytes(), PAIRED_TYPES, SPLIT_COLORS, SPLIT_LABELS, AISetupScreen(), AISetupScreenProps (+31 more)

### Community 23 - "DevTrainPanel.tsx"
Cohesion: 0.13
Nodes (26): analyzeError(), AppliedEditInfo, calculateAffectedLines(), ChatSession, deleteScript(), devProgressPercent(), DevTrainErrorModal(), DevTrainErrorModalProps (+18 more)

### Community 24 - "plugins/datasetCompat.ts"
Cohesion: 0.17
Nodes (24): DatasetCompatBadge(), DatasetCompatBadgeProps, AnalysisPreviewProps, checkDatasetCompat(), COMPAT_PLUGINS, analysisToCheckInput(), CompatLevel, DATASET_TYPE_LABELS (+16 more)

### Community 25 - "MessageProtocol"
Cohesion: 0.06
Nodes (34): ABC, Any, core/config.py – TrainingConfig ================================ Gemeinsame…, Hilfsmethode: Plugin-spezifischen Wert aus plugin_config holen., TrainingConfig, build_training_arguments(), cap_eval_dataset(), classification_scores() (+26 more)

### Community 26 - "SynapseBuilder.tsx"
Cohesion: 0.18
Nodes (23): kv(), AffectedNodeInfo, applyShapeHighlightsToEdges(), applyShapeHighlightsToNodes(), buildShapeAgentPrompt(), buildShapeUserGuide(), clearShapeHighlights(), collectAffectedNodeIds() (+15 more)

### Community 27 - "CanvasPlugin"
Cohesion: 0.12
Nodes (8): Optimizer, CanvasPlugin, Any, Module, TrainingConfig, Lädt vorherige Gewichte + Optimizer-State für echten Resume., W1: Scheduler-Instanz basierend auf IR-Konfiguration. steps_per_epoch wird für…, Fix 1.3: Speichert vollständigen IR + optimizer_state_dict für Inference-Reload…

### Community 28 - "compilerOptions"
Cohesion: 0.08
Nodes (24): DOM, DOM.Iterable, ES2020, src, compilerOptions, allowImportingTsExtensions, allowSyntheticDefaultImports, esModuleInterop (+16 more)

### Community 29 - "FloatingAICoach.tsx"
Cohesion: 0.05
Nodes (69): AICoachOpenDetail, onOpenAICoach(), openAICoach(), APP_OVERVIEW, Bilingual, buildCoachSystemPrompt(), buildPageContext(), CoachAction (+61 more)

### Community 30 - "ModelLibrary.tsx"
Cohesion: 0.11
Nodes (19): CanvasModelMetadata, exportCanvasNetworkToModelLibrary(), updateCanvasNetworkModel(), CanvasGraphIR, generateModelConfigFromGraph(), LayerConfig, ModelGraphConfig, CanvasDesign (+11 more)

### Community 31 - "yolo/manifest.json"
Cohesion: 0.08
Nodes (23): box_loss, cls_loss, mAP50, mAP50-95, pascal_voc, yolo, yolo11, yolo_bbox (+15 more)

### Community 32 - "Settings.tsx"
Cohesion: 0.10
Nodes (30): callAI(), CallAIOptions, callAnthropic(), callOllama(), callOpenAICompat(), ChatRole, effectiveMaxTokens(), requireEnabled() (+22 more)

### Community 33 - "usePageContext"
Cohesion: 0.13
Nodes (21): PageId, Settings(), useStoredTickets(), formatBytes(), formatDate(), formatDuration(), getFileIcon(), ModelVersion (+13 more)

### Community 34 - "canvas/manifest.json"
Cohesion: 0.09
Nodes (22): csv_loader, image_loader, numpy>=1.24.0,<2.0.0, pandas>=2.0.0, parquet_loader, pyarrow>=14.0.0, torch>=2.0.0, class (+14 more)

### Community 35 - "image_classification/manifest.json"
Cohesion: 0.09
Nodes (22): efficientnet_b0, efficientnet_b4, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet50, top5_accuracy, vit_b_16 (+14 more)

### Community 36 - ".status"
Cohesion: 0.09
Nodes (12): _audio_in(), _class_dirs(), Plugin, Path, TrainingConfig, _class_dirs(), _images_in(), Plugin (+4 more)

### Community 37 - "OpenLibraryModal.tsx"
Cohesion: 0.13
Nodes (22): addToLocalLibrary(), AUTHOR_KEY(), DuplicateNameError(), FRAMEWORKS, getLocalKey(), getStoredAuthorName(), LibraryScript, MODEL_TYPE_COLORS (+14 more)

### Community 38 - "nodeTypes.ts"
Cohesion: 0.07
Nodes (27): dragState, categoryColors, icons, NodeLibrary(), NodeLibraryProps, handleStyle(), SynapseNodeComponent, SynapseNodeData (+19 more)

### Community 39 - "SynapseAIPanel.tsx"
Cohesion: 0.24
Nodes (15): ChatMessage, AgentResumeState, AgentStep, errorActionBtnStyle, formatChatDate(), iconBtnStyle(), SynapseAIPanel(), SynapseAIPanelProps (+7 more)

### Community 40 - "analysis_manager.rs"
Cohesion: 0.36
Nodes (21): analysis_dir(), check_version_ownership(), db_path(), delete_ai_analysis_report(), get_ai_analysis_report(), get_training_full_data(), get_training_logs(), get_training_metrics() (+13 more)

### Community 41 - "ImageClassificationPlugin"
Cohesion: 0.13
Nodes (13): _build_backbone(), _build_transforms(), _classification_scores(), _freeze_base(), ImageClassificationPlugin, _load_datasets(), Any, DataLoader (+5 more)

### Community 42 - "dev_trainer.rs"
Cohesion: 0.27
Nodes (20): DevProcEntry, DevTrainingRefs, get_python_path(), kill_process_tree(), parse_loss_from_line(), registry_clear(), registry_set_pid(), registry_start() (+12 more)

### Community 43 - "supported_architectures"
Cohesion: 0.06
Nodes (33): beit, convnext, convnextv2, cvt, deit, dinat, efficientnet, levit (+25 more)

### Community 44 - "devDependencies"
Cohesion: 0.11
Nodes (19): autoprefixer, jsdom, devDependencies, autoprefixer, jsdom, tailwindcss, @tauri-apps/cli, @testing-library/jest-dom (+11 more)

### Community 45 - "dependencies"
Cohesion: 0.11
Nodes (19): lucide-react, dependencies, lucide-react, react, react-dom, @tauri-apps/api, @tauri-apps/plugin-dialog, @tauri-apps/plugin-fs (+11 more)

### Community 46 - "TrainingConsole.tsx"
Cohesion: 0.12
Nodes (14): CanvasInferenceResult, CanvasInferenceTab(), CanvasModelInfo, parseInputString(), pct(), Props, DatasetOption, ExportModal() (+6 more)

### Community 47 - "Plugin"
Cohesion: 0.19
Nodes (7): _detect_columns(), Plugin, Path, TrainingConfig, Sequenzklassifikations-Plugin für XLM-RoBERTa & ähnliche Encoder., Prüft Modell-Architektur und initialisiert Tokenizer., Erkennt automatisch Text- und Label-Spalte.

### Community 48 - "ui.test.tsx"
Cohesion: 0.09
Nodes (22): hfEncoderPlugin, DatasetProgress, HFEncoderTestPlugin(), PredRow, TopPred, HFEncoderTrainPlugin(), BASE_TEST_PROPS, ListenerMap (+14 more)

### Community 49 - "synapseAgent.ts"
Cohesion: 0.12
Nodes (25): AgentRunOptions, AgentRunResult, buildFixSystem(), buildPlanSystem(), executeBatch(), extractRetryDelayMs(), getFixMaxTokens(), getPlanMaxTokens() (+17 more)

### Community 50 - "codeGenerator.ts"
Cohesion: 0.15
Nodes (24): buildCompactGraphSummary(), genDatasetCode(), genDynamicForwardLines(), generateTrainingScript(), genForwardLine(), genForwardLineWithInputs(), genInitLine(), getCategory() (+16 more)

### Community 51 - "graph-shape-validation.ts"
Cohesion: 0.24
Nodes (16): detectCycles(), getSynapseNodeType(), isShapeCompatible(), LAYER_SHAPE_METADATA, nodeParams(), outputFeatureSize(), printValidationReport(), SHAPE_FLOW_CHECKED_TARGETS (+8 more)

### Community 52 - "definitions"
Cohesion: 0.12
Nodes (16): definitions, Number, PermissionEntry, ShellScopeEntryAllowedArg, Target, Value, anyOf, description (+8 more)

### Community 53 - "definitions"
Cohesion: 0.12
Nodes (16): definitions, Number, PermissionEntry, ShellScopeEntryAllowedArg, ShellScopeEntryAllowedArgs, Value, anyOf, description (+8 more)

### Community 54 - "supported_architectures"
Cohesion: 0.07
Nodes (28): ast, audio-spectrogram-transformer, hubert, sew, sew-d, unispeech, unispeech-sat, wav2vec2 (+20 more)

### Community 55 - "DatasetUpload.tsx"
Cohesion: 0.09
Nodes (26): CoachPromptOptions, PageContextInput, AIAnalysisReport, DatasetCard(), DatasetCardProps, DatasetStructureGuide(), DatasetUpload(), DeleteDialog() (+18 more)

### Community 56 - "image-classification/index.ts"
Cohesion: 0.24
Nodes (8): containsToken(), detectImageClassification(), imageClassificationPlugin, NON_CLASSIFIER_IMAGE_TOKENS, TORCHVISION_TOKENS, ImageClassificationTestPlugin(), Prediction, ImageClassificationTrainPlugin()

### Community 57 - "types.ts"
Cohesion: 0.10
Nodes (26): DatasetFileManagerProps, DatasetInfo, audioClassificationPlugin, AudioTestPlugin(), AudioTrainPlugin(), canvasPlugin, DatasetType, PairingStatus (+18 more)

### Community 58 - "properties"
Cohesion: 0.15
Nodes (13): properties, Identifier, default, description, type, description, oneOf, type (+5 more)

### Community 59 - ".error"
Cohesion: 0.23
Nodes (4): Any, Path, TrainingConfig, YOLOPlugin

### Community 60 - "PowerState"
Cohesion: 0.25
Nodes (13): allow_sleep(), disable_prevent_sleep(), enable_prevent_sleep(), get_prevent_sleep_status(), PowerState, prevent_sleep(), Child, Default (+5 more)

### Community 61 - "yolo/index.ts"
Cohesion: 0.21
Nodes (9): detectYOLO(), yoloPlugin, Detection, InferenceResult, YOLOTestPlugin(), DatasetInfo, formatBytes(), YOLO_PRESETS (+1 more)

### Community 62 - "main.rs"
Cohesion: 0.37
Nodes (12): Database, clear_config(), force_quit_app(), get_app_data_dir(), load_config(), open_path_in_finder(), read_model_config(), AppHandle (+4 more)

### Community 63 - "properties"
Cohesion: 0.13
Nodes (15): properties, default, description, type, type, array, null, description (+7 more)

### Community 64 - "permissions"
Cohesion: 0.17
Nodes (11): core:default, dialog:default, fs:default, main, os:default, shell:allow-open, description, identifier (+3 more)

### Community 65 - "DynamicGraphModule"
Cohesion: 0.11
Nodes (22): _apply_dense_initializer(), DynamicGraphModule, CanvasGraphIR, Module, Tensor, DynamicGraphModule — runtime DAG forward from Canvas IR., UI-Param 'initializer' des Dense-Nodes auf nn.Linear anwenden., Runtime nn.Module built from Canvas Graph IR. Exposes .layers ModuleDict for… (+14 more)

### Community 66 - "permissions"
Cohesion: 0.29
Nodes (7): $ref, description, items, type, uniqueItems, items, permissions

### Community 67 - "supported_architectures"
Cohesion: 0.08
Nodes (24): bart, blenderbot, longt5, m2m_100, marian, mbart, mt5, pegasus (+16 more)

### Community 68 - "hf-encoder/detect.ts"
Cohesion: 0.25
Nodes (8): containsToken(), detectHFEncoder(), HF_ENCODER_SUPPORTED_MODEL_TYPES, modelNameSegment(), NON_ENCODER_TOKENS, SUPPORTED_MODEL_TYPES, detectXLMRoberta(), XLM_ROBERTA_ARCHITECTURES

### Community 69 - "webviews"
Cohesion: 0.20
Nodes (10): type, webviews, windows, items, description, items, type, description (+2 more)

### Community 70 - "webviews"
Cohesion: 0.20
Nodes (10): type, webviews, windows, items, description, items, type, description (+2 more)

### Community 71 - "generate-icons.py"
Cohesion: 0.29
Nodes (9): create_base_icon(), generate_icns(), generate_ico(), generate_png_icons(), main(), Erstellt das Basis-Icon mit RGBA (Alpha), Generiert PNG Icons in RGBA, Generiert Windows .ico (+1 more)

### Community 72 - "auth.rs"
Cohesion: 0.33
Nodes (9): ApiKeyValidation, CredentialRequest, CredentialResponse, Option, Result, String, test_api_key_format_validation(), test_empty_password() (+1 more)

### Community 73 - "compilerOptions"
Cohesion: 0.22
Nodes (8): vite.config.ts, compilerOptions, allowSyntheticDefaultImports, composite, module, moduleResolution, skipLibCheck, include

### Community 74 - "codeEdits.ts"
Cohesion: 0.19
Nodes (18): anchorReplace(), applyAllEdits(), applyEdit(), collapseSpaces(), findLineWindow(), nonEmptyLines(), normalizeTabs(), parseEdits() (+10 more)

### Community 75 - "synapseAgentTools.ts"
Cohesion: 0.28
Nodes (8): AgentToolExecutor, createToolExecutor(), GraphMutationEvent, parsePosition(), PLAN_TOOLS, ToolExecutorContext, ToolExecutorHandle, waitForVisualMutation()

### Community 76 - "ModelConfig"
Cohesion: 0.22
Nodes (15): AUDIO_MODEL_TYPES, detectAudioClassification(), NON_CLASSIFIER, SUPPORTED, detectHFImageClassification(), HF_IMAGE_MODEL_TYPES, NON_CLASSIFIER, SUPPORTED (+7 more)

### Community 77 - "CapabilityRemote"
Cohesion: 0.22
Nodes (9): description, properties, required, type, CapabilityRemote, urls, urls, description (+1 more)

### Community 78 - "CapabilityRemote"
Cohesion: 0.22
Nodes (9): description, properties, required, type, CapabilityRemote, urls, urls, description (+1 more)

### Community 79 - "ModelServer"
Cohesion: 0.14
Nodes (9): detect_modality(), emit(), emit_error(), main(), ModelServer, Path, Klassennamen aus label_mapping.json (id2label ODER classes) bzw. config.json.…, Laedt eine Audiodatei als Mono-Wellenform in der Modell-Samplerate. Gleiche… (+1 more)

### Community 80 - "test_engine/plugins/seq_classification/manifest.json"
Cohesion: 0.22
Nodes (8): class, description, entry, input_type, name, output_format, task_type, version

### Community 81 - "errorReport.ts"
Cohesion: 0.17
Nodes (12): App(), cat(), t(), analyzeError(), ErrorRecoveryPanel(), APP_ERROR_ENDPOINT, AppErrorReport, installGlobalErrorReporting() (+4 more)

### Community 82 - "AIKnowledgeBaseSmart.ts"
Cohesion: 0.25
Nodes (4): AI_SYSTEM_PROMPT_WITH_INSTRUCTIONS, KNOWLEDGE_SECTIONS, KNOWLEDGE_TOC, KnowledgeSection

### Community 83 - "permissions"
Cohesion: 0.17
Nodes (12): $ref, array, null, description, items, type, uniqueItems, description (+4 more)

### Community 84 - "api_config.rs"
Cohesion: 0.38
Nodes (5): get_api_base_url(), is_local_dev_api(), String, test_endpoint_construction(), validate_credentials()

### Community 85 - "package.json"
Cohesion: 0.33
Nodes (5): license, name, private, type, version

### Community 86 - "scripts"
Cohesion: 0.33
Nodes (6): scripts, build, dev, preview, tauri:build, tauri:dev

### Community 87 - "Capability"
Cohesion: 0.33
Nodes (6): description, required, type, Capability, identifier, permissions

### Community 88 - "Capability"
Cohesion: 0.33
Nodes (6): description, required, type, Capability, identifier, permissions

### Community 89 - "desktop-schema.json"
Cohesion: 0.40
Nodes (4): anyOf, description, $schema, title

### Community 90 - "macOS-schema.json"
Cohesion: 0.40
Nodes (4): anyOf, description, $schema, title

### Community 91 - "ModelManager.tsx"
Cohesion: 0.08
Nodes (31): AnalysisPreview(), checkHfModelSupport(), DeleteConfirmDialog(), DeleteDialogProps, formatBytes(), formatDate(), formatDownloads(), HuggingFaceImportPanel() (+23 more)

### Community 93 - "local"
Cohesion: 0.50
Nodes (4): default, description, type, local

### Community 94 - "local"
Cohesion: 0.50
Nodes (4): default, description, type, local

### Community 96 - "run_dataset_classification"
Cohesion: 0.20
Nodes (10): collect_class_files(), load_label_names(), Any, Path, Gemeinsame Bausteine für klassifizierende Test-Plugins (Bild, Audio). Beide…, Klassennamen aus label_mapping.json oder id2label des Modells., Sammelt Dateien samt erwarteter Klasse. Unterstützt das Trainingslayout (Ordner…, Durchläuft ein Dataset und schreibt Ergebnisse + Kennzahlen. (+2 more)

### Community 97 - "ShellScopeEntryAllowedArgs"
Cohesion: 0.67
Nodes (3): ShellScopeEntryAllowedArgs, anyOf, description

### Community 98 - "Target"
Cohesion: 0.67
Nodes (3): Target, description, oneOf

### Community 131 - "Plugin"
Cohesion: 0.22
Nodes (6): Plugin, Any, Path, TestConfig, Wählt die richtige Datei aus einem Dataset-Verzeichnis. WICHTIG: Bei…, Lädt Samples aus JSON/JSONL/CSV/Parquet in eine einheitliche Struktur.

### Community 132 - "NotificationContext.tsx"
Cohesion: 0.18
Nodes (10): Notification, NotificationAction, NotificationContainerProps, NotificationContext, NotificationContextType, NotificationItem(), NotificationItemProps, NotificationOptions (+2 more)

### Community 133 - "DiffViewer.tsx"
Cohesion: 0.27
Nodes (9): AutoAction, CodeEdit, AiMessage, AiMessage, countAddedRemoved(), countLines(), DiffViewer(), DiffViewerProps (+1 more)

### Community 134 - "TestConfig"
Cohesion: 0.20
Nodes (6): Any, core/config.py – TestConfig, Hilfsmethode: Plugin-spezifischen Wert aus plugin_config holen., TestConfig, Test-Plugin: Seq2Seq-Inferenz (Zusammenfassung, Übersetzung, Umformung)., plugins/seq_classification/plugin.py – Test/Inferenz-Plugin…

### Community 135 - "Plugin"
Cohesion: 0.22
Nodes (5): Plugin, Any, Path, TestConfig, Test-Plugin: Audioklassifikation mit einem trainierten HuggingFace-Modell.

### Community 136 - "Plugin"
Cohesion: 0.22
Nodes (5): Plugin, Any, Path, TestConfig, Test-Plugin: Bildklassifikation mit einem trainierten HuggingFace-Modell.

### Community 137 - "Plugin"
Cohesion: 0.29
Nodes (4): Plugin, Any, Path, TestConfig

### Community 138 - "TestProtocol"
Cohesion: 0.33
Nodes (4): Any, core/protocol.py – JSON-Kommunikationsprotokoll für die Test-Engine, TestProtocol, main()

### Community 139 - "test_engine/plugins/audio_classification/manifest.json"
Cohesion: 0.22
Nodes (8): class, description, entry, input_type, name, output_format, task_type, version

### Community 140 - "test_engine/plugins/hf_image_classification/manifest.json"
Cohesion: 0.22
Nodes (8): class, description, entry, input_type, name, output_format, task_type, version

### Community 141 - "test_engine/plugins/seq2seq/manifest.json"
Cohesion: 0.22
Nodes (8): class, description, entry, input_type, name, output_format, task_type, version

### Community 142 - "DatasetInfo"
Cohesion: 0.67
Nodes (4): DevTestPanelProps, DevTrainPanelProps, DatasetInfo, ModelInfo

### Community 143 - "Identifier"
Cohesion: 0.67
Nodes (3): Identifier, description, oneOf

## Ambiguous Edges - Review These
- `SynapseBuilder()` → `applyAutoFix()`  [AMBIGUOUS]
  src/components/synapse/ai/INTEGRATION_GUIDE.md · relation: calls

## Knowledge Gaps
- **678 isolated node(s):** `name`, `private`, `license`, `version`, `type` (+673 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `SynapseBuilder()` and `applyAutoFix()`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `useLanguage()` connect `useLanguage` to `TrainingPanel.tsx`, `DevTestPanel.tsx`, `DiffViewer.tsx`, `Dashboard.tsx`, `SynapseAICoachPanel.tsx`, `AnalysisPanel.tsx`, `TrainingDashboard.tsx`, `LaboratoryPanel.tsx`, `DevTrainPanel.tsx`, `plugins/datasetCompat.ts`, `SynapseBuilder.tsx`, `FloatingAICoach.tsx`, `ModelLibrary.tsx`, `Settings.tsx`, `usePageContext`, `OpenLibraryModal.tsx`, `nodeTypes.ts`, `SynapseAIPanel.tsx`, `TrainingConsole.tsx`, `DatasetUpload.tsx`, `codeEdits.ts`, `errorReport.ts`, `ModelManager.tsx`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `SynapseAICoach` connect `SynapseAICoachPanel.tsx` to `Manager (Cloudflare Worker + D1)`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `Auto-Fix Pipeline (App-Fehler → Manager → Triage-Agent → Review → Merge)` connect `Manager (Cloudflare Worker + D1)` to `SynapseAICoachPanel.tsx`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **What connects `name`, `private`, `license` to the rest of the system?**
  _678 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `dataset_manager.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.08767696267696268 - nodes in this community are weakly interconnected._
- **Should `training_manager.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.06222222222222222 - nodes in this community are weakly interconnected._