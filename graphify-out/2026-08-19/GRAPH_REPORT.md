# Graph Report - desktop-app  (2026-08-18)

## Corpus Check
- 193 files · ~326,328 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2305 nodes · 5220 edges · 131 communities (116 shown, 15 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.68)
- Token cost: 238,547 input · 0 output

## Community Hubs (Navigation)
- Dataset Manager (Rust)
- Training Manager (Rust)
- AI Coach Panel Integration
- SQLite Database Layer (Rust)
- Canvas IR Data Loading (Python)
- Dev Auto Mode & Code Edits
- Synapse Auto-Fix Helper
- App Shell & Dialogs
- Test Engine Config & Plugins
- Auto-Fix Pipeline Docs
- Analysis Panel (React)
- Laboratory Manager (Rust)
- Tauri App Config
- Plugin Dependency Detection (Rust)
- Test Manager (Rust)
- Model Version Manager (Rust)
- Seq-Classification Model Manifest
- Training Dashboard & Error Reporting
- Train Engine Entry Point (Python)
- Model Manager (Rust)
- Laboratory Panel (React)
- Canvas Inference Server (Python)
- First Launch Setup UI
- Dev Train Panel (React)
- Dataset Compatibility Checks
- Train Engine Plugin Base & Config
- Synapse Shape Diagnostics
- Canvas Train Plugin (Python)
- TypeScript Compiler Config
- AI Coach Context Builder
- Canvas Graph IR & Model Library
- YOLO Plugin Manifest
- AI Provider Client (Multi-backend)
- Version Manager & Language Context
- Canvas Plugin Manifest & Deps
- Image Classification Manifest
- Floating AI Coach (React)
- Open Library Modal
- Synapse Node Library & Drag State
- Synapse AI Panel (Chat)
- Analysis Manager (Rust)
- Image Classification Plugin (Python)
- Dev Trainer Process (Rust)
- Synapse Node Types
- package.json devDependencies
- package.json dependencies
- Canvas Inference & Training Console
- Seq-Classification Plugin (Python)
- HF-Encoder Plugin UI Tests
- Synapse AI Agent
- Synapse Code Generator
- Synapse Graph Shape Validation
- Desktop Tauri Schema Definitions
- macOS Tauri Schema Definitions
- AI Coach Events & Context Menu
- Settings Panel (React)
- Image-Classification Plugin (TS)
- Plugin Registry & Index
- Desktop Schema Properties
- YOLO Plugin (Python)
- Power Manager (Rust)
- YOLO Plugin (TS)
- Main Entry & App Commands (Rust)
- macOS Schema Properties
- Tauri Capabilities Config
- Canvas Model Bridge
- macOS Schema Array Definitions
- Synapse Debug Logger
- HF-Encoder Model Detection
- Desktop Schema Webview/Window Defs
- macOS Schema Webview/Window Defs
- Icon Generation Script
- Auth & API Key Validation (Rust)
- Vite Node TS Config
- AI Coach Prompt Builder
- Synapse Agent Tools
- XLM-RoBERTa Test Plugin
- Desktop Schema Capability Remote
- macOS Schema Capability Remote
- Test Engine Model Server (Python)
- Test Engine Seq-Classification Manifest
- Train Engine Message Protocol
- AI Knowledge Base (Smart)
- Desktop Schema Permissions
- API Config (Rust)
- package.json Metadata
- package.json Scripts
- Desktop Schema Capability
- macOS Schema Capability
- Desktop Schema Root
- macOS Schema Root
- Model Selector (React)
- Synapse Ops Parser
- Desktop Schema Local Flag
- macOS Schema Local Flag
- Desktop Schema Identifier
- Desktop Schema Shell Args
- macOS Schema Target
- package.json postcss
- package.json testing-library
- package.json @types/node
- package.json @types/react
- package.json @types/react-dom
- package.json vitest
- Placeholder Icon Script
- Canvas Plugin Package Init
- Vite Env Types
- Cargo.toml Package Name
- App Icon 128x128@2x
- App Icon 128x128
- App Icon 32x32
- App Favicon SVG

## God Nodes (most connected - your core abstractions)
1. `useLanguage()` - 123 edges
2. `AppState` - 63 edges
3. `Database` - 38 edges
4. `useTheme()` - 36 edges
5. `SynapseBuilderInner()` - 32 edges
6. `useNotification()` - 31 edges
7. `usePageContext()` - 25 edges
8. `split_dataset_in_half()` - 23 edges
9. `useAISettings()` - 23 edges
10. `dateLocale()` - 22 edges

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
- **Auto-Fix Pipeline: error capture, triage agent, and Manager review loop** — claude_automation_setup_auto_fix_pipeline, claude_automation_setup_manager, claude_commands_triage_errors_triage_agent, github_workflows_auto_triage_job [INFERRED 0.85]
- **Release workflow job dependency chain (create-release → build-tauri → generate-latest-json → notify-release)** — github_workflows_release_create_release_job, github_workflows_release_build_tauri_job, github_workflows_release_generate_latest_json_job, github_workflows_release_notify_release_job [EXTRACTED 1.00]
- **Synapse AI Coach system components (core logic, sub-modules, and UI panel)** — src_components_synapse_ai_synapseaicoach_synapseaicoach, src_components_synapse_ai_synapseaicoach_errorparser, src_components_synapse_ai_synapseaicoach_graphanalyzer, src_components_synapse_ai_synapseaicoach_graphautofixer, src_components_synapse_ai_synapseaicoachpanel_synapseaicoachpanel [EXTRACTED 1.00]

## Communities (131 total, 15 thin omitted)

### Community 0 - "Dataset Manager (Rust)"
Cohesion: 0.10
Nodes (99): Client, add_files_to_dataset(), alte_ascii_namen_werden_erkannt(), analyze_dataset_path(), check_basename_pairing(), collect_extensions(), collect_files(), collect_files_recursive() (+91 more)

### Community 1 - "Training Manager (Rust)"
Cohesion: 0.06
Nodes (73): CanvasInferenceResult, CanvasModelInfo, CanvasNetworkResult, check_training_requirements(), copy_dir(), create_canvas_network_model(), create_version(), default_lora_mods() (+65 more)

### Community 2 - "AI Coach Panel Integration"
Cohesion: 0.04
Nodes (73): CoachCommand, CoachConfigPatch, consumePendingCoachCommand(), getRecommendedParams(), onApplyCoachConfig(), onCoachCommand(), runCoachCommand(), setRecommendedParams() (+65 more)

### Community 3 - "SQLite Database Layer (Rust)"
Cohesion: 0.08
Nodes (36): Connection, Database, Dataset, get_data_directory(), get_database_path(), Model, ModelVersion, Option (+28 more)

### Community 4 - "Canvas IR Data Loading (Python)"
Cohesion: 0.05
Nodes (48): get_dataloaders(), _infer_input_features(), CanvasGraphIR, DataLoader, DataLoaders for Canvas IR training., _apply_dense_initializer(), DynamicGraphModule, CanvasGraphIR (+40 more)

### Community 5 - "Dev Auto Mode & Code Edits"
Cohesion: 0.07
Nodes (56): AutoAction, AutoMode, buildAutoSystemPrompt(), parseAutoAction(), anchorReplace(), applyAllEdits(), applyEdit(), CodeEdit (+48 more)

### Community 6 - "Synapse Auto-Fix Helper"
Cohesion: 0.06
Nodes (38): adjustParamsFix(), applyAutoFix(), AutoFixResult, insertBridgeFix(), TODO: Implementiere topologisches Sorting der Nodes, TODO: Toast anzeigen, TODO: Error Toast anzeigen, removeNodeFix() (+30 more)

### Community 7 - "App Shell & Dialogs"
Cohesion: 0.05
Nodes (47): ApiKeyValidation, CloseDialogProps, NOTE: synapse_sessions_${userId} NICHT löschen - Sessions sind User-Daten und…, UserData, Dashboard(), DashboardProps, UserData, View (+39 more)

### Community 8 - "Test Engine Config & Plugins"
Cohesion: 0.08
Nodes (25): Any, core/config.py – TestConfig, Hilfsmethode: Plugin-spezifischen Wert aus plugin_config holen., TestConfig, Any, core/protocol.py – JSON-Kommunikationsprotokoll für die Test-Engine, TestProtocol, Plugin (+17 more)

### Community 9 - "Auto-Fix Pipeline Docs"
Cohesion: 0.05
Nodes (46): ANTHROPIC_API_KEY (GitHub Actions secret), app_errors D1 table (extended: error_group, triage_status, occurrences, screen), Auto-Fix Pipeline (App-Fehler → Manager → Triage-Agent → Review → Merge), AUTOMATION_SECRET (Cloudflare secret), Nightly cron routine (0 1 * * * ≈ 03:00 Berlin, or manual workflow_dispatch), FrameTrain Auto-Fix Pipeline Setup Guide, src/utils/errorReport.ts – installGlobalErrorReporting(), fix_proposals D1 table (+38 more)

### Community 10 - "Analysis Panel (React)"
Cohesion: 0.09
Nodes (37): AnalysisPanel(), buildFullContext(), AnalysisPanelProps, asFiniteNumber(), BigLossChart(), BOOLEAN_PARAM_KEYS, buildAnalysisSystemPrompt(), ChatMessage (+29 more)

### Community 11 - "Laboratory Manager (Rust)"
Cohesion: 0.15
Nodes (38): BufWriter, ChildStdin, Receiver, get_canvas_server_path(), get_model_server_path(), get_python_path(), get_version_info(), get_version_path() (+30 more)

### Community 12 - "Tauri App Config"
Cohesion: 0.05
Nodes (39): **, https://github.com/FrameSphere/FrameTrain-App/releases/latest/download/latest.json, icons/128x128@2x.png, icons/128x128.png, icons/32x32.png, icons/icon.icns, icons/icon.ico, python (+31 more)

### Community 13 - "Plugin Dependency Detection (Rust)"
Cohesion: 0.16
Nodes (37): GpuInfo, check_dependency_status(), check_first_launch(), check_package_installed(), DependencyStatus, detect_gpu(), find_valid_python(), get_available_plugins() (+29 more)

### Community 14 - "Test Manager (Rust)"
Cohesion: 0.20
Nodes (37): default_mode(), default_task_type(), export_hard_examples(), get_active_test_job(), get_current_test(), get_models_dir(), get_python_path(), get_test_engine_path() (+29 more)

### Community 15 - "Model Version Manager (Rust)"
Cohesion: 0.16
Nodes (27): ModelInfo, calculate_directory_size(), copy_dir_recursive_export(), Database, delete_model_version(), export_model_version(), get_version_path_for_ui(), list_model_versions() (+19 more)

### Community 16 - "Seq-Classification Model Manifest"
Cohesion: 0.06
Nodes (35): albert, bert, camembert, csv, deberta, deberta-v2, distilbert, electra (+27 more)

### Community 17 - "Training Dashboard & Error Reporting"
Cohesion: 0.09
Nodes (30): PageId, App(), analyzeError(), BigLossChart(), ConfigSummary(), ErrorCategory, ErrorRecoveryPanel(), EVENT_COLORS (+22 more)

### Community 18 - "Train Engine Entry Point (Python)"
Cohesion: 0.08
Nodes (16): handle_exception(), is_oom(), is_torch_ecosystem_conflict(), load_plugin(), LoggingTee, main(), Orchestrator, Exception (+8 more)

### Community 19 - "Model Manager (Rust)"
Cohesion: 0.20
Nodes (34): calc_speed_and_eta(), calculate_dir_size(), cleanup_incomplete_download(), copy_dir_recursive(), delete_model(), detect_model_type(), download_huggingface_model(), get_directory_size() (+26 more)

### Community 20 - "Laboratory Panel (React)"
Cohesion: 0.10
Nodes (30): AnalysisView(), deleteSession(), extractLabelField(), extractTextField(), getDisplayText(), getSideInfo(), LABEL_KEYS, LaboratoryPanel() (+22 more)

### Community 21 - "Canvas Inference Server (Python)"
Cohesion: 0.11
Nodes (21): CanvasInferenceServer, emit(), emit_error(), main(), _find_model_pt(), load_canvas_model(), _load_ir_from_checkpoint(), _load_ir_from_metadata() (+13 more)

### Community 22 - "First Launch Setup UI"
Cohesion: 0.10
Nodes (26): DatasetStructureGuide(), DeleteDialog(), AISetupScreen(), AISetupScreenProps, DependencyStatus, FirstLaunchSetup(), GpuInfo, InstallProgress (+18 more)

### Community 23 - "Dev Train Panel (React)"
Cohesion: 0.12
Nodes (28): DevTestPanelProps, analyzeError(), AppliedEditInfo, calculateAffectedLines(), ChatSession, deleteScript(), DevTrainErrorModal(), DevTrainErrorModalProps (+20 more)

### Community 24 - "Dataset Compatibility Checks"
Cohesion: 0.19
Nodes (22): DatasetCompatBadge(), DatasetCompatBadgeProps, checkDatasetCompat(), COMPAT_PLUGINS, analysisToCheckInput(), CompatLevel, DATASET_TYPE_LABELS, DatasetAnalysis (+14 more)

### Community 25 - "Train Engine Plugin Base & Config"
Cohesion: 0.10
Nodes (10): ABC, Any, core/config.py – TrainingConfig ================================ Gemeinsame…, Hilfsmethode: Plugin-spezifischen Wert aus plugin_config holen., TrainingConfig, Any, TrainingConfig, core/plugin_base.py – Abstrakte Basisklasse für Trainings-Plugins (+2 more)

### Community 26 - "Synapse Shape Diagnostics"
Cohesion: 0.17
Nodes (23): AffectedNodeInfo, applyShapeHighlightsToEdges(), applyShapeHighlightsToNodes(), buildShapeAgentPrompt(), buildShapeUserGuide(), clearShapeHighlights(), collectAffectedNodeIds(), getAffectedNodes() (+15 more)

### Community 27 - "Canvas Train Plugin (Python)"
Cohesion: 0.13
Nodes (8): Optimizer, CanvasPlugin, Any, Module, TrainingConfig, Lädt vorherige Gewichte + Optimizer-State für echten Resume., W1: Scheduler-Instanz basierend auf IR-Konfiguration. steps_per_epoch wird für…, Fix 1.3: Speichert vollständigen IR + optimizer_state_dict für Inference-Reload…

### Community 28 - "TypeScript Compiler Config"
Cohesion: 0.08
Nodes (24): DOM, DOM.Iterable, ES2020, src, compilerOptions, allowImportingTsExtensions, allowSyntheticDefaultImports, esModuleInterop (+16 more)

### Community 29 - "AI Coach Context Builder"
Cohesion: 0.09
Nodes (24): APP_OVERVIEW, Bilingual, buildPageContext(), coercePatchFromRecord(), coerceSettable(), ContextLine, FALSE_WORDS, formatConfigPatch() (+16 more)

### Community 30 - "Canvas Graph IR & Model Library"
Cohesion: 0.11
Nodes (21): buildCanvasGraphIR(), CANVAS_IR_VERSION, getCategory(), getParams(), IRDataSpec, IREdge, IRNode, IRTrainingSpec (+13 more)

### Community 31 - "YOLO Plugin Manifest"
Cohesion: 0.08
Nodes (23): box_loss, cls_loss, mAP50, mAP50-95, pascal_voc, yolo, yolo11, yolo_bbox (+15 more)

### Community 32 - "AI Provider Client (Multi-backend)"
Cohesion: 0.15
Nodes (20): callAI(), CallAIOptions, callAnthropic(), callOllama(), callOpenAICompat(), ChatRole, requireEnabled(), withResponseLanguage() (+12 more)

### Community 33 - "Version Manager & Language Context"
Cohesion: 0.12
Nodes (22): CoachPromptOptions, PageContextInput, AIAnalysisReport, LanguageSelectScreenProps, formatBytes(), formatDate(), formatDuration(), getFileIcon() (+14 more)

### Community 34 - "Canvas Plugin Manifest & Deps"
Cohesion: 0.09
Nodes (22): csv_loader, image_loader, numpy>=1.24.0,<2.0.0, pandas>=2.0.0, parquet_loader, pyarrow>=14.0.0, torch>=2.0.0, class (+14 more)

### Community 35 - "Image Classification Manifest"
Cohesion: 0.09
Nodes (22): efficientnet_b0, efficientnet_b4, folder_class, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet50, top5_accuracy (+14 more)

### Community 36 - "Floating AI Coach (React)"
Cohesion: 0.14
Nodes (21): CoachAction, hasPageKnowledge(), Chat, createFreshChat(), darkenHex(), FloatingAICoach(), FloatingAICoachProps, formatPageContextTitle() (+13 more)

### Community 37 - "Open Library Modal"
Cohesion: 0.13
Nodes (22): addToLocalLibrary(), AUTHOR_KEY(), DuplicateNameError(), FRAMEWORKS, getLocalKey(), getStoredAuthorName(), LibraryScript, MODEL_TYPE_COLORS (+14 more)

### Community 38 - "Synapse Node Library & Drag State"
Cohesion: 0.10
Nodes (12): dragState, categoryColors, icons, NodeLibrary(), NodeLibraryProps, NODE_CATEGORIES, NodeDefinition, ParamDefinition (+4 more)

### Community 39 - "Synapse AI Panel (Chat)"
Cohesion: 0.17
Nodes (19): ChatMessage, AgentResumeState, AgentStep, errorActionBtnStyle, formatChatDate(), iconBtnStyle(), SynapseAIPanel(), SynapseAIPanelProps (+11 more)

### Community 40 - "Analysis Manager (Rust)"
Cohesion: 0.36
Nodes (21): analysis_dir(), check_version_ownership(), db_path(), delete_ai_analysis_report(), get_ai_analysis_report(), get_training_full_data(), get_training_logs(), get_training_metrics() (+13 more)

### Community 41 - "Image Classification Plugin (Python)"
Cohesion: 0.15
Nodes (11): _build_backbone(), _build_transforms(), _freeze_base(), ImageClassificationPlugin, _load_datasets(), Any, DataLoader, Module (+3 more)

### Community 42 - "Dev Trainer Process (Rust)"
Cohesion: 0.27
Nodes (20): DevProcEntry, DevTrainingRefs, get_python_path(), kill_process_tree(), parse_loss_from_line(), registry_clear(), registry_set_pid(), registry_start() (+12 more)

### Community 43 - "Synapse Node Types"
Cohesion: 0.15
Nodes (16): handleStyle(), SynapseNodeComponent, SynapseNodeData, calcInputHandleTop(), calcOutputHandleTop(), CATEGORY_META, CATEGORY_NODES, LAYOUT (+8 more)

### Community 44 - "package.json devDependencies"
Cohesion: 0.11
Nodes (19): autoprefixer, jsdom, devDependencies, autoprefixer, jsdom, tailwindcss, @tauri-apps/cli, @testing-library/jest-dom (+11 more)

### Community 45 - "package.json dependencies"
Cohesion: 0.11
Nodes (19): lucide-react, dependencies, lucide-react, react, react-dom, @tauri-apps/api, @tauri-apps/plugin-dialog, @tauri-apps/plugin-fs (+11 more)

### Community 46 - "Canvas Inference & Training Console"
Cohesion: 0.12
Nodes (14): CanvasInferenceResult, CanvasInferenceTab(), CanvasModelInfo, parseInputString(), pct(), Props, DatasetOption, ExportModal() (+6 more)

### Community 47 - "Seq-Classification Plugin (Python)"
Cohesion: 0.19
Nodes (8): _detect_columns(), Plugin, Path, TrainingConfig, plugins/seq_classification/plugin.py ===================================== XLM-…, Sequenzklassifikations-Plugin für XLM-RoBERTa & ähnliche Encoder., Prüft Modell-Architektur und initialisiert Tokenizer., Erkennt automatisch Text- und Label-Spalte.

### Community 48 - "HF-Encoder Plugin UI Tests"
Cohesion: 0.13
Nodes (12): DatasetProgress, HFEncoderTestPlugin(), PredRow, TopPred, HFEncoderTrainPlugin(), BASE_TEST_PROPS, ListenerMap, listeners (+4 more)

### Community 49 - "Synapse AI Agent"
Cohesion: 0.21
Nodes (16): AgentRunOptions, AgentRunResult, buildFixSystem(), buildPlanSystem(), executeBatch(), extractRetryDelayMs(), getFixMaxTokens(), getPlanMaxTokens() (+8 more)

### Community 50 - "Synapse Code Generator"
Cohesion: 0.29
Nodes (15): buildCompactGraphSummary(), genDatasetCode(), genDynamicForwardLines(), generateTrainingScript(), genForwardLine(), genForwardLineWithInputs(), genInitLine(), getCategory() (+7 more)

### Community 51 - "Synapse Graph Shape Validation"
Cohesion: 0.26
Nodes (15): detectCycles(), getSynapseNodeType(), isShapeCompatible(), LAYER_SHAPE_METADATA, nodeParams(), outputFeatureSize(), SHAPE_FLOW_CHECKED_TARGETS, SHAPE_RESETTING_TYPES (+7 more)

### Community 52 - "Desktop Tauri Schema Definitions"
Cohesion: 0.12
Nodes (16): definitions, Number, PermissionEntry, ShellScopeEntryAllowedArg, Target, Value, anyOf, description (+8 more)

### Community 53 - "macOS Tauri Schema Definitions"
Cohesion: 0.12
Nodes (16): definitions, Number, PermissionEntry, ShellScopeEntryAllowedArg, ShellScopeEntryAllowedArgs, Value, anyOf, description (+8 more)

### Community 54 - "AI Coach Events & Context Menu"
Cohesion: 0.20
Nodes (12): AICoachOpenDetail, onOpenAICoach(), openAICoach(), AppContextMenu(), MenuState, NAV_ITEMS, Row, Section (+4 more)

### Community 55 - "Settings Panel (React)"
Cohesion: 0.15
Nodes (13): kv(), CommunityNameErrorModal(), InstallProgress, Settings(), SettingsProps, SettingsTab, STATUS_COLOR, StoredTicket (+5 more)

### Community 56 - "Image-Classification Plugin (TS)"
Cohesion: 0.22
Nodes (10): DatasetInfo, DatasetType, PairingStatus, imageClassificationPlugin, ImageClassificationTestPlugin(), Prediction, ImageClassificationTrainPlugin(), DatasetInfo (+2 more)

### Community 57 - "Plugin Registry & Index"
Cohesion: 0.25
Nodes (8): canvasPlugin, hfEncoderPlugin, KNOWN_UNSUPPORTED, ModelDetectionInfo, PLUGINS, REQUIRED_FIELDS, ModelPlugin, xlmRobertaPlugin

### Community 58 - "Desktop Schema Properties"
Cohesion: 0.13
Nodes (15): properties, default, description, type, type, array, null, description (+7 more)

### Community 59 - "YOLO Plugin (Python)"
Cohesion: 0.24
Nodes (4): Any, Path, TrainingConfig, YOLOPlugin

### Community 60 - "Power Manager (Rust)"
Cohesion: 0.25
Nodes (13): allow_sleep(), disable_prevent_sleep(), enable_prevent_sleep(), get_prevent_sleep_status(), PowerState, prevent_sleep(), Child, Default (+5 more)

### Community 61 - "YOLO Plugin (TS)"
Cohesion: 0.19
Nodes (10): ModelConfig, detectYOLO(), yoloPlugin, Detection, InferenceResult, YOLOTestPlugin(), DatasetInfo, formatBytes() (+2 more)

### Community 62 - "Main Entry & App Commands (Rust)"
Cohesion: 0.37
Nodes (12): Database, clear_config(), force_quit_app(), get_app_data_dir(), load_config(), open_path_in_finder(), read_model_config(), AppHandle (+4 more)

### Community 63 - "macOS Schema Properties"
Cohesion: 0.15
Nodes (13): properties, Identifier, default, description, type, description, oneOf, type (+5 more)

### Community 64 - "Tauri Capabilities Config"
Cohesion: 0.17
Nodes (11): core:default, dialog:default, fs:default, main, os:default, shell:allow-open, description, identifier (+3 more)

### Community 65 - "Canvas Model Bridge"
Cohesion: 0.21
Nodes (7): CanvasModelMetadata, exportCanvasNetworkToModelLibrary(), updateCanvasNetworkModel(), CanvasGraphIR, generateModelConfigFromGraph(), LayerConfig, ModelGraphConfig

### Community 66 - "macOS Schema Array Definitions"
Cohesion: 0.17
Nodes (12): $ref, array, null, description, items, type, uniqueItems, description (+4 more)

### Community 67 - "Synapse Debug Logger"
Cohesion: 0.24
Nodes (9): buildEntry(), DebugCallEntry, DebugHandle, estimateTokens(), findDuplicates(), flush(), isEnabled(), logLines (+1 more)

### Community 68 - "HF-Encoder Model Detection"
Cohesion: 0.25
Nodes (8): containsToken(), detectHFEncoder(), HF_ENCODER_SUPPORTED_MODEL_TYPES, modelNameSegment(), NON_ENCODER_TOKENS, SUPPORTED_MODEL_TYPES, detectXLMRoberta(), XLM_ROBERTA_ARCHITECTURES

### Community 69 - "Desktop Schema Webview/Window Defs"
Cohesion: 0.20
Nodes (10): type, webviews, windows, items, description, items, type, description (+2 more)

### Community 70 - "macOS Schema Webview/Window Defs"
Cohesion: 0.20
Nodes (10): type, webviews, windows, items, description, items, type, description (+2 more)

### Community 71 - "Icon Generation Script"
Cohesion: 0.29
Nodes (9): create_base_icon(), generate_icns(), generate_ico(), generate_png_icons(), main(), Erstellt das Basis-Icon mit RGBA (Alpha), Generiert PNG Icons in RGBA, Generiert Windows .ico (+1 more)

### Community 72 - "Auth & API Key Validation (Rust)"
Cohesion: 0.33
Nodes (9): ApiKeyValidation, CredentialRequest, CredentialResponse, Option, Result, String, test_api_key_format_validation(), test_empty_password() (+1 more)

### Community 73 - "Vite Node TS Config"
Cohesion: 0.22
Nodes (8): vite.config.ts, compilerOptions, allowSyntheticDefaultImports, composite, module, moduleResolution, skipLibCheck, include

### Community 74 - "AI Coach Prompt Builder"
Cohesion: 0.31
Nodes (9): buildCoachSystemPrompt(), commandLabel(), linkTargetLabel(), navTargetLabel(), pageKnowledge(), pick(), toolsProtocol(), applyCoachConfig() (+1 more)

### Community 75 - "Synapse Agent Tools"
Cohesion: 0.28
Nodes (8): AgentToolExecutor, createToolExecutor(), GraphMutationEvent, parsePosition(), PLAN_TOOLS, ToolExecutorContext, ToolExecutorHandle, waitForVisualMutation()

### Community 76 - "XLM-RoBERTa Test Plugin"
Cohesion: 0.25
Nodes (8): DatasetProgress, DatasetResults, formatBytes(), PredRow, SingleResult, TopPred, UnlistenFn, XLMRobertaTestPlugin()

### Community 77 - "Desktop Schema Capability Remote"
Cohesion: 0.22
Nodes (9): description, properties, required, type, CapabilityRemote, urls, urls, description (+1 more)

### Community 78 - "macOS Schema Capability Remote"
Cohesion: 0.22
Nodes (9): description, properties, required, type, CapabilityRemote, urls, urls, description (+1 more)

### Community 79 - "Test Engine Model Server (Python)"
Cohesion: 0.42
Nodes (4): emit(), emit_error(), main(), ModelServer

### Community 80 - "Test Engine Seq-Classification Manifest"
Cohesion: 0.22
Nodes (8): class, description, entry, input_type, name, output_format, task_type, version

### Community 81 - "Train Engine Message Protocol"
Cohesion: 0.39
Nodes (3): MessageProtocol, Any, YOLO Object Detection Plugin — task_type: 'detect

### Community 82 - "AI Knowledge Base (Smart)"
Cohesion: 0.25
Nodes (4): AI_SYSTEM_PROMPT_WITH_INSTRUCTIONS, KNOWLEDGE_SECTIONS, KNOWLEDGE_TOC, KnowledgeSection

### Community 83 - "Desktop Schema Permissions"
Cohesion: 0.29
Nodes (7): $ref, description, items, type, uniqueItems, items, permissions

### Community 84 - "API Config (Rust)"
Cohesion: 0.38
Nodes (5): get_api_base_url(), is_local_dev_api(), String, test_endpoint_construction(), validate_credentials()

### Community 85 - "package.json Metadata"
Cohesion: 0.33
Nodes (5): license, name, private, type, version

### Community 86 - "package.json Scripts"
Cohesion: 0.33
Nodes (6): scripts, build, dev, preview, tauri:build, tauri:dev

### Community 87 - "Desktop Schema Capability"
Cohesion: 0.33
Nodes (6): description, required, type, Capability, identifier, permissions

### Community 88 - "macOS Schema Capability"
Cohesion: 0.33
Nodes (6): description, required, type, Capability, identifier, permissions

### Community 89 - "Desktop Schema Root"
Cohesion: 0.40
Nodes (4): anyOf, description, $schema, title

### Community 90 - "macOS Schema Root"
Cohesion: 0.40
Nodes (4): anyOf, description, $schema, title

### Community 91 - "Model Selector (React)"
Cohesion: 0.67
Nodes (3): ModelSelectorProps, State, DetectionResult

### Community 93 - "Desktop Schema Local Flag"
Cohesion: 0.50
Nodes (4): default, description, type, local

### Community 94 - "macOS Schema Local Flag"
Cohesion: 0.50
Nodes (4): default, description, type, local

### Community 96 - "Desktop Schema Identifier"
Cohesion: 0.67
Nodes (3): Identifier, description, oneOf

### Community 97 - "Desktop Schema Shell Args"
Cohesion: 0.67
Nodes (3): ShellScopeEntryAllowedArgs, anyOf, description

### Community 98 - "macOS Schema Target"
Cohesion: 0.67
Nodes (3): Target, description, oneOf

## Ambiguous Edges - Review These
- `SynapseBuilder()` → `applyAutoFix()`  [AMBIGUOUS]
  src/components/synapse/ai/INTEGRATION_GUIDE.md · relation: calls

## Knowledge Gaps
- **564 isolated node(s):** `name`, `private`, `license`, `version`, `type` (+559 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `SynapseBuilder()` and `applyAutoFix()`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `useLanguage()` connect `First Launch Setup UI` to `AI Coach Panel Integration`, `Dev Auto Mode & Code Edits`, `Synapse Auto-Fix Helper`, `App Shell & Dialogs`, `Analysis Panel (React)`, `Training Dashboard & Error Reporting`, `Laboratory Panel (React)`, `Dev Train Panel (React)`, `Dataset Compatibility Checks`, `Synapse Shape Diagnostics`, `Canvas Graph IR & Model Library`, `AI Provider Client (Multi-backend)`, `Version Manager & Language Context`, `Floating AI Coach (React)`, `Open Library Modal`, `Synapse Node Library & Drag State`, `Synapse AI Panel (Chat)`, `Canvas Inference & Training Console`, `AI Coach Events & Context Menu`, `Settings Panel (React)`, `AI Coach Prompt Builder`, `Model Selector (React)`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `AppState` connect `Dataset Manager (Rust)` to `Training Manager (Rust)`, `SQLite Database Layer (Rust)`, `Analysis Manager (Rust)`, `Model Version Manager (Rust)`, `Model Manager (Rust)`, `Main Entry & App Commands (Rust)`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `SynapseAICoach` connect `Synapse Auto-Fix Helper` to `Auto-Fix Pipeline Docs`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **What connects `name`, `private`, `license` to the rest of the system?**
  _564 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dataset Manager (Rust)` be split into smaller, more focused modules?**
  _Cohesion score 0.09522031366691561 - nodes in this community are weakly interconnected._
- **Should `Training Manager (Rust)` be split into smaller, more focused modules?**
  _Cohesion score 0.06287363430220573 - nodes in this community are weakly interconnected._