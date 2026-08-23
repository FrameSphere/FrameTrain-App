# Graph Report - desktop-app  (2026-08-23)

## Corpus Check
- 234 files · ~348,939 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2729 nodes · 6074 edges · 168 communities (153 shown, 15 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a60105a2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- dataset_manager.rs
- training_manager.rs
- TrainingPanel.tsx
- Database
- ir.py
- DevTestPanel.tsx
- autoFixHelper.ts
- LanguageContext.tsx
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
- TrainPlugin
- SynapseBuilder.tsx
- CanvasPlugin
- compilerOptions
- FloatingAICoach.tsx
- ModelLibrary.tsx
- yolo/manifest.json
- AISettingsContext.tsx
- Settings.tsx
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
- graphIR.ts
- Plugin
- ui.test.tsx
- synapseAgent.ts
- codeGenerator.ts
- graph-shape-validation.ts
- definitions
- definitions
- supported_architectures
- MessageProtocol
- image-classification/index.ts
- registry.ts
- properties
- YOLOPlugin
- PowerState
- types.ts
- main.rs
- properties
- permissions
- DynamicGraphModule
- permissions
- supported_architectures
- xlm-roberta/TestPlugin.tsx
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
- SynapseAICoachPanel.tsx
- AIKnowledgeBaseSmart.ts
- permissions
- api_config.rs
- package.json
- scripts
- Capability
- Capability
- desktop-schema.json
- macOS-schema.json
- DatasetUpload.tsx
- synapseOps.ts
- local
- local
- run_dataset_classification
- TestPluginProps
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
- AppContextMenu.tsx
- DiffViewer.tsx
- TestConfig
- Plugin
- Plugin
- Plugin
- TestProtocol
- test_engine/plugins/audio_classification/manifest.json
- test_engine/plugins/hf_image_classification/manifest.json
- test_engine/plugins/seq2seq/manifest.json
- CanvasGraphIR
- train_engine/plugins/audio_classification/plugin.py
- synapseDebugLogger.ts
- aiClient.ts
- make_plugin
- python_env.rs
- synapseGraphContext.ts
- canvasModelBridge.ts
- parse_ir
- layer_factory.py
- shape_propagate.py
- devScriptStorage.ts
- NotificationContext.tsx
- GradientChatInput.tsx
- Identifier
- Target
- DevTestPanel
- CanvasInferenceTab.tsx
- PageContext.tsx
- DatasetInfo

## God Nodes (most connected - your core abstractions)
1. `useLanguage()` - 123 edges
2. `AppState` - 63 edges
3. `Database` - 38 edges
4. `useTheme()` - 36 edges
5. `SynapseBuilderInner()` - 32 edges
6. `useNotification()` - 31 edges
7. `detect_dataset_type()` - 28 edges
8. `usePageContext()` - 25 edges
9. `MessageProtocol` - 24 edges
10. `split_dataset_in_half()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Auto-Fix Pipeline (App-Fehler → Manager → Triage-Agent → Review → Merge)` --semantically_similar_to--> `SynapseAICoach`  [INFERRED] [semantically similar]
  .claude/AUTOMATION_SETUP.md → src/components/synapse/ai/SynapseAICoach.ts
- `build-tauri job (matrix: macOS aarch64, Ubuntu, Windows)` --references--> `index.html – Vite/Tauri Entry Point`  [INFERRED]
  .github/workflows/release.yml → index.html
- `Synapse AI Coach Integration Guide` --references--> `SynapseBuilder()`  [EXTRACTED]
  src/components/synapse/ai/INTEGRATION_GUIDE.md → src/components/synapse/SynapseBuilder.tsx
- `SynapseBuilder()` --shares_data_with--> `currentModelConfig state`  [EXTRACTED]
  src/components/synapse/SynapseBuilder.tsx → src/components/synapse/ai/INTEGRATION_GUIDE.md
- `Synapse AI Coach System README` --references--> `SynapseAICoach`  [EXTRACTED]
  src/components/synapse/ai/README_AI_COACH.md → src/components/synapse/ai/SynapseAICoach.ts

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Release workflow job dependency chain (create-release → build-tauri → generate-latest-json → notify-release)** — github_workflows_release_create_release_job, github_workflows_release_build_tauri_job, github_workflows_release_generate_latest_json_job, github_workflows_release_notify_release_job [EXTRACTED 1.00]
- **Synapse AI Coach system components (core logic, sub-modules, and UI panel)** — src_components_synapse_ai_synapseaicoach_synapseaicoach, src_components_synapse_ai_synapseaicoach_errorparser, src_components_synapse_ai_synapseaicoach_graphanalyzer, src_components_synapse_ai_synapseaicoach_graphautofixer, src_components_synapse_ai_synapseaicoachpanel_synapseaicoachpanel [EXTRACTED 1.00]
- **Auto-Fix Pipeline: error capture, triage agent, and Manager review loop** — claude_automation_setup_auto_fix_pipeline, claude_automation_setup_manager, claude_commands_triage_errors_triage_agent, github_workflows_auto_triage_job [INFERRED 0.85]

## Communities (168 total, 15 thin omitted)

### Community 0 - "dataset_manager.rs"
Cohesion: 0.07
Nodes (131): Client, Drop, add_files_to_dataset(), alte_ascii_namen_werden_erkannt(), analyze_dataset_path(), build_nested(), canonical_split_name(), check_basename_pairing() (+123 more)

### Community 1 - "training_manager.rs"
Cohesion: 0.06
Nodes (73): CanvasInferenceResult, CanvasModelInfo, CanvasNetworkResult, check_training_requirements(), copy_dir(), create_canvas_network_model(), create_version(), default_lora_mods() (+65 more)

### Community 2 - "TrainingPanel.tsx"
Cohesion: 0.09
Nodes (29): openAICoach(), applyCoachConfig(), CoachCommand, CoachConfigPatch, consumePendingCoachConfig(), getRecommendedParams(), onApplyCoachConfig(), clampNumber() (+21 more)

### Community 3 - "Database"
Cohesion: 0.08
Nodes (36): Connection, Database, Dataset, get_data_directory(), get_database_path(), Model, ModelVersion, Option (+28 more)

### Community 4 - "ir.py"
Cohesion: 0.25
Nodes (10): IREdge, IRNode, IRTrainingSpec, Canvas Graph IR — parsed from config.canvas_graph JSON., build_model_from_graph(), CanvasGraphIR, build_model_from_graph — runtime nn.Module from IR., Canvas Model Training Plugin — Runtime Graph IR Trainiert Modelle aus… (+2 more)

### Community 5 - "DevTestPanel.tsx"
Cohesion: 0.11
Nodes (27): AutoMode, buildAutoSystemPrompt(), parseAutoAction(), extractFullPythonCode(), removeEditBlocks(), AppliedEditInfo, ChatSession, CodeAISidebar() (+19 more)

### Community 6 - "autoFixHelper.ts"
Cohesion: 0.10
Nodes (18): adjustParamsFix(), applyAutoFix(), AutoFixResult, insertBridgeFix(), TODO: Implementiere topologisches Sorting der Nodes, TODO: Toast anzeigen, TODO: Error Toast anzeigen, removeNodeFix() (+10 more)

### Community 7 - "LanguageContext.tsx"
Cohesion: 0.07
Nodes (31): ApiKeyValidation, CloseDialogProps, NOTE: synapse_sessions_${userId} NICHT löschen - Sessions sind User-Daten und…, UserData, Dashboard(), DashboardProps, UserData, View (+23 more)

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
Cohesion: 0.16
Nodes (37): BufWriter, ChildStdin, Receiver, get_canvas_server_path(), get_model_server_path(), get_python_path(), get_version_info(), get_version_path() (+29 more)

### Community 12 - "tauri.conf.json"
Cohesion: 0.05
Nodes (39): **, https://github.com/FrameSphere/FrameTrain-App/releases/latest/download/latest.json, icons/128x128@2x.png, icons/128x128.png, icons/32x32.png, icons/icon.icns, icons/icon.ico, python (+31 more)

### Community 13 - "plugin_commands.rs"
Cohesion: 0.13
Nodes (40): GpuInfo, check_dependency_status(), check_first_launch(), check_package_installed(), DependencyStatus, detect_gpu(), find_valid_python(), get_available_plugins() (+32 more)

### Community 14 - "test_manager.rs"
Cohesion: 0.20
Nodes (36): default_mode(), default_task_type(), export_hard_examples(), get_active_test_job(), get_current_test(), get_models_dir(), get_python_path(), get_test_engine_path() (+28 more)

### Community 15 - "String"
Cohesion: 0.16
Nodes (27): ModelInfo, calculate_directory_size(), copy_dir_recursive_export(), Database, delete_model_version(), export_model_version(), get_version_path_for_ui(), list_model_versions() (+19 more)

### Community 16 - "supported_architectures"
Cohesion: 0.06
Nodes (35): albert, bert, camembert, deberta, deberta-v2, distilbert, electra, ernie (+27 more)

### Community 17 - "TrainingDashboard.tsx"
Cohesion: 0.09
Nodes (30): App(), firstUsableLoss(), lossImprovementPct(), LossPointLike, cat(), t(), analyzeError(), BigLossChart() (+22 more)

### Community 18 - "train_engine.py"
Cohesion: 0.08
Nodes (16): handle_exception(), is_oom(), is_torch_ecosystem_conflict(), load_plugin(), LoggingTee, main(), Orchestrator, Exception (+8 more)

### Community 19 - "model_manager.rs"
Cohesion: 0.18
Nodes (37): calc_speed_and_eta(), calculate_dir_size(), calculate_dir_size_inner(), cleanup_incomplete_download(), copy_dir_recursive(), delete_model(), detect_model_type(), download_huggingface_model() (+29 more)

### Community 20 - "LaboratoryPanel.tsx"
Cohesion: 0.10
Nodes (29): hasOpenQuote(), joinQuotedLines(), parseDelimitedRows(), splitDelimitedLine(), AnalysisView(), deleteSession(), extractLabelField(), extractTextField() (+21 more)

### Community 21 - "load_canvas_model"
Cohesion: 0.11
Nodes (21): CanvasInferenceServer, emit(), emit_error(), main(), _find_model_pt(), load_canvas_model(), _load_ir_from_checkpoint(), _load_ir_from_metadata() (+13 more)

### Community 22 - "useLanguage"
Cohesion: 0.08
Nodes (37): DatasetFileManager(), FileInfo, formatBytes(), PAIRED_TYPES, SPLIT_COLORS, SPLIT_LABELS, DatasetStructureGuide(), DeleteDialog() (+29 more)

### Community 23 - "DevTrainPanel.tsx"
Cohesion: 0.13
Nodes (27): analyzeError(), AppliedEditInfo, calculateAffectedLines(), ChatSession, deleteScript(), devProgressPercent(), DevTrainErrorModal(), DevTrainErrorModalProps (+19 more)

### Community 24 - "plugins/datasetCompat.ts"
Cohesion: 0.19
Nodes (22): DatasetCompatBadge(), DatasetCompatBadgeProps, checkDatasetCompat(), COMPAT_PLUGINS, analysisToCheckInput(), CompatLevel, DATASET_TYPE_LABELS, DatasetAnalysis (+14 more)

### Community 25 - "TrainPlugin"
Cohesion: 0.09
Nodes (14): ABC, Any, core/config.py – TrainingConfig ================================ Gemeinsame…, Hilfsmethode: Plugin-spezifischen Wert aus plugin_config holen., TrainingConfig, Any, TrainingConfig, core/plugin_base.py – Abstrakte Basisklasse für Trainings-Plugins (+6 more)

### Community 26 - "SynapseBuilder.tsx"
Cohesion: 0.14
Nodes (28): kv(), AffectedNodeInfo, applyShapeHighlightsToEdges(), applyShapeHighlightsToNodes(), buildShapeAgentPrompt(), buildShapeUserGuide(), clearShapeHighlights(), collectAffectedNodeIds() (+20 more)

### Community 27 - "CanvasPlugin"
Cohesion: 0.14
Nodes (6): CanvasPlugin, Any, Module, TrainingConfig, Lädt vorherige Gewichte + Optimizer-State für echten Resume., Fix 1.3: Speichert vollständigen IR + optimizer_state_dict für Inference-Reload…

### Community 28 - "compilerOptions"
Cohesion: 0.08
Nodes (24): DOM, DOM.Iterable, ES2020, src, compilerOptions, allowImportingTsExtensions, allowSyntheticDefaultImports, esModuleInterop (+16 more)

### Community 29 - "FloatingAICoach.tsx"
Cohesion: 0.05
Nodes (61): AICoachOpenDetail, onOpenAICoach(), APP_OVERVIEW, Bilingual, buildCoachSystemPrompt(), buildPageContext(), CoachAction, CoachPromptOptions (+53 more)

### Community 30 - "ModelLibrary.tsx"
Cohesion: 0.21
Nodes (11): CanvasDesign, deleteBtn, deleteModelDesign(), formatDate(), iconBtn, loadBtn, loadModelDesign(), ModelLibrary() (+3 more)

### Community 31 - "yolo/manifest.json"
Cohesion: 0.08
Nodes (23): box_loss, cls_loss, mAP50, mAP50-95, pascal_voc, yolo, yolo11, yolo_bbox (+15 more)

### Community 32 - "AISettingsContext.tsx"
Cohesion: 0.23
Nodes (11): PROVIDER_META, ProviderMeta, AIAssistantSettingsPanel(), AIProvider, AISettings, AISettingsContext, AISettingsContextType, DEFAULT_SETTINGS (+3 more)

### Community 33 - "Settings.tsx"
Cohesion: 0.10
Nodes (25): CommunityNameErrorModal(), InstallProgress, Settings(), SettingsProps, SettingsTab, STATUS_COLOR, StoredTicket, SupportMessage (+17 more)

### Community 34 - "canvas/manifest.json"
Cohesion: 0.09
Nodes (22): csv_loader, image_loader, numpy>=1.24.0,<2.0.0, pandas>=2.0.0, parquet_loader, pyarrow>=14.0.0, torch>=2.0.0, class (+14 more)

### Community 35 - "image_classification/manifest.json"
Cohesion: 0.09
Nodes (22): efficientnet_b0, efficientnet_b4, mobilenet_v3_large, mobilenet_v3_small, resnet18, resnet50, top5_accuracy, vit_b_16 (+14 more)

### Community 36 - ".status"
Cohesion: 0.15
Nodes (6): _class_dirs(), Plugin, Path, TrainingConfig, Plugin, TrainingConfig

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
Cohesion: 0.14
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

### Community 46 - "graphIR.ts"
Cohesion: 0.10
Nodes (17): buildCanvasGraphIR(), CANVAS_IR_VERSION, getCategory(), getParams(), IRDataSpec, IREdge, IRNode, IRTrainingSpec (+9 more)

### Community 47 - "Plugin"
Cohesion: 0.19
Nodes (7): _detect_columns(), Plugin, Path, TrainingConfig, Sequenzklassifikations-Plugin für XLM-RoBERTa & ähnliche Encoder., Prüft Modell-Architektur und initialisiert Tokenizer., Erkennt automatisch Text- und Label-Spalte.

### Community 48 - "ui.test.tsx"
Cohesion: 0.14
Nodes (12): DatasetProgress, HFEncoderTestPlugin(), PredRow, TopPred, HFEncoderTrainPlugin(), BASE_TEST_PROPS, ListenerMap, listeners (+4 more)

### Community 49 - "synapseAgent.ts"
Cohesion: 0.22
Nodes (15): AgentRunOptions, AgentRunResult, buildFixSystem(), buildPlanSystem(), executeBatch(), extractRetryDelayMs(), getFixMaxTokens(), getPlanMaxTokens() (+7 more)

### Community 50 - "codeGenerator.ts"
Cohesion: 0.29
Nodes (15): buildCompactGraphSummary(), genDatasetCode(), genDynamicForwardLines(), generateTrainingScript(), genForwardLine(), genForwardLineWithInputs(), genInitLine(), getCategory() (+7 more)

### Community 51 - "graph-shape-validation.ts"
Cohesion: 0.24
Nodes (16): detectCycles(), getSynapseNodeType(), isShapeCompatible(), LAYER_SHAPE_METADATA, nodeParams(), outputFeatureSize(), printValidationReport(), SHAPE_FLOW_CHECKED_TARGETS (+8 more)

### Community 52 - "definitions"
Cohesion: 0.12
Nodes (16): definitions, Number, PermissionEntry, ShellScopeEntryAllowedArg, ShellScopeEntryAllowedArgs, Value, anyOf, description (+8 more)

### Community 53 - "definitions"
Cohesion: 0.12
Nodes (16): definitions, Number, PermissionEntry, ShellScopeEntryAllowedArg, ShellScopeEntryAllowedArgs, Value, anyOf, description (+8 more)

### Community 54 - "supported_architectures"
Cohesion: 0.07
Nodes (28): ast, audio-spectrogram-transformer, hubert, sew, sew-d, unispeech, unispeech-sat, wav2vec2 (+20 more)

### Community 55 - "MessageProtocol"
Cohesion: 0.11
Nodes (20): build_training_arguments(), cap_eval_dataset(), classification_scores(), device_name(), _epoch_number(), final_metrics(), optimizer_name(), progress_callback() (+12 more)

### Community 56 - "image-classification/index.ts"
Cohesion: 0.24
Nodes (8): containsToken(), detectImageClassification(), imageClassificationPlugin, NON_CLASSIFIER_IMAGE_TOKENS, TORCHVISION_TOKENS, ImageClassificationTestPlugin(), Prediction, ImageClassificationTrainPlugin()

### Community 57 - "registry.ts"
Cohesion: 0.15
Nodes (13): audioClassificationPlugin, AudioTrainPlugin(), canvasPlugin, hfEncoderPlugin, KNOWN_UNSUPPORTED, ModelDetectionInfo, PLUGINS, TEXT_DOMAIN_KEYS (+5 more)

### Community 58 - "properties"
Cohesion: 0.13
Nodes (15): properties, default, description, type, type, array, null, description (+7 more)

### Community 59 - "YOLOPlugin"
Cohesion: 0.14
Nodes (8): Any, Path, TrainingConfig, Summiert box/cls/dfl-Loss eines Praefixes ('train/' oder 'val/')., Laufender Trainings-Loss (box + cls + dfl) der aktuellen Epoche., Zaehlt die Bilder je Split anhand der dataset.yaml. Die Analyse-Seite zeigte…, Waehlt die Startgewichte. Ohne diesen Schritt wurde immer 'yolov8n.pt' geladen…, YOLOPlugin

### Community 60 - "PowerState"
Cohesion: 0.25
Nodes (13): allow_sleep(), disable_prevent_sleep(), enable_prevent_sleep(), get_prevent_sleep_status(), PowerState, prevent_sleep(), Child, Default (+5 more)

### Community 61 - "types.ts"
Cohesion: 0.15
Nodes (15): DatasetFileManagerProps, DatasetInfo, DatasetType, PairingStatus, DatasetInfo, TrainPluginProps, detectYOLO(), yoloPlugin (+7 more)

### Community 62 - "main.rs"
Cohesion: 0.37
Nodes (12): Database, clear_config(), force_quit_app(), get_app_data_dir(), load_config(), open_path_in_finder(), read_model_config(), AppHandle (+4 more)

### Community 63 - "properties"
Cohesion: 0.15
Nodes (13): properties, Identifier, default, description, type, description, oneOf, type (+5 more)

### Community 64 - "permissions"
Cohesion: 0.17
Nodes (11): core:default, dialog:default, fs:default, main, os:default, shell:allow-open, description, identifier (+3 more)

### Community 65 - "DynamicGraphModule"
Cohesion: 0.13
Nodes (15): _apply_dense_initializer(), DynamicGraphModule, CanvasGraphIR, Module, Tensor, DynamicGraphModule — runtime DAG forward from Canvas IR., UI-Param 'initializer' des Dense-Nodes auf nn.Linear anwenden., Runtime nn.Module built from Canvas Graph IR. Exposes .layers ModuleDict for… (+7 more)

### Community 66 - "permissions"
Cohesion: 0.17
Nodes (12): $ref, array, null, description, items, type, uniqueItems, description (+4 more)

### Community 67 - "supported_architectures"
Cohesion: 0.08
Nodes (24): bart, blenderbot, longt5, m2m_100, marian, mbart, mt5, pegasus (+16 more)

### Community 68 - "xlm-roberta/TestPlugin.tsx"
Cohesion: 0.13
Nodes (16): containsToken(), detectHFEncoder(), HF_ENCODER_SUPPORTED_MODEL_TYPES, modelNameSegment(), NON_ENCODER_TOKENS, SUPPORTED_MODEL_TYPES, detectXLMRoberta(), XLM_ROBERTA_ARCHITECTURES (+8 more)

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
Cohesion: 0.20
Nodes (17): anchorReplace(), applyAllEdits(), applyEdit(), collapseSpaces(), findLineWindow(), nonEmptyLines(), normalizeTabs(), parseEdits() (+9 more)

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

### Community 81 - "SynapseAICoachPanel.tsx"
Cohesion: 0.14
Nodes (12): DiagnosticIssue, ErrorParser, FixSuggestion, GraphAnalyzer, GraphDiagnosis, TODO: Implement node reordering, SynapseAICoach, TrainingAnalyzer (+4 more)

### Community 82 - "AIKnowledgeBaseSmart.ts"
Cohesion: 0.25
Nodes (4): AI_SYSTEM_PROMPT_WITH_INSTRUCTIONS, KNOWLEDGE_SECTIONS, KNOWLEDGE_TOC, KnowledgeSection

### Community 83 - "permissions"
Cohesion: 0.29
Nodes (7): $ref, description, items, type, uniqueItems, items, permissions

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

### Community 91 - "DatasetUpload.tsx"
Cohesion: 0.06
Nodes (50): consumePendingCoachCommand(), onCoachCommand(), AnalysisPreview(), AnalysisPreviewProps, DatasetCard(), DatasetCardProps, DatasetUpload(), FilterOptions (+42 more)

### Community 93 - "local"
Cohesion: 0.50
Nodes (4): default, description, type, local

### Community 94 - "local"
Cohesion: 0.50
Nodes (4): default, description, type, local

### Community 96 - "run_dataset_classification"
Cohesion: 0.20
Nodes (10): collect_class_files(), load_label_names(), Any, Path, Gemeinsame Bausteine für klassifizierende Test-Plugins (Bild, Audio). Beide…, Klassennamen aus label_mapping.json oder id2label des Modells., Sammelt Dateien samt erwarteter Klasse. Unterstützt das Trainingslayout (Ordner…, Durchläuft ein Dataset und schreibt Ergebnisse + Kennzahlen. (+2 more)

### Community 97 - "TestPluginProps"
Cohesion: 0.22
Nodes (9): AudioTestPlugin(), GenericTestPanel(), GenericTestPanelProps, TopPred, hfImageClassificationPlugin, HFImageTestPlugin(), HFImageTrainPlugin(), Seq2SeqTestPlugin() (+1 more)

### Community 98 - "Target"
Cohesion: 0.67
Nodes (3): Target, description, oneOf

### Community 131 - "Plugin"
Cohesion: 0.22
Nodes (6): Plugin, Any, Path, TestConfig, Wählt die richtige Datei aus einem Dataset-Verzeichnis. WICHTIG: Bei…, Lädt Samples aus JSON/JSONL/CSV/Parquet in eine einheitliche Struktur.

### Community 132 - "AppContextMenu.tsx"
Cohesion: 0.22
Nodes (11): AppContextMenu(), MenuState, NAV_ITEMS, Row, Section, collectContextMenuActions(), ContextMenuAction, Provider (+3 more)

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

### Community 142 - "CanvasGraphIR"
Cohesion: 0.20
Nodes (8): Optimizer, get_dataloaders(), _infer_input_features(), CanvasGraphIR, DataLoader, DataLoaders for Canvas IR training., CanvasGraphIR, W1: Scheduler-Instanz basierend auf IR-Konfiguration. steps_per_epoch wird für…

### Community 143 - "train_engine/plugins/audio_classification/plugin.py"
Cohesion: 0.20
Nodes (6): _audio_in(), _class_dirs(), Plugin, Path, TrainingConfig, Audio Classification (HuggingFace) ================================== Schliesst…

### Community 150 - "synapseDebugLogger.ts"
Cohesion: 0.24
Nodes (10): buildEntry(), DebugCallEntry, DebugHandle, debugLogRequest(), estimateTokens(), findDuplicates(), flush(), isEnabled() (+2 more)

### Community 151 - "aiClient.ts"
Cohesion: 0.31
Nodes (10): callAI(), CallAIOptions, callAnthropic(), callOllama(), callOpenAICompat(), ChatRole, effectiveMaxTokens(), requireEnabled() (+2 more)

### Community 152 - "make_plugin"
Cohesion: 0.11
Nodes (10): YOLO Object Detection Plugin — task_type: 'detect, FakeTrainer, make_plugin(), MetricsTest, Prueft Startgewichte, Metriken und Split-Groessen des YOLO-Plugins.…, Regression: Analyse zeigte "n_train 0 / n_val 0" trotz 463/116 Bildern., Nachbau der Ultralytics-Trainer-Attribute, die der Callback liest., Regression: Train Loss stand im Trainingsdialog dauerhaft auf 0.0000. Der… (+2 more)

### Community 153 - "python_env.rs"
Cohesion: 0.38
Nodes (10): Candidate, candidates(), fallback(), parse_version(), resolve_python(), resolve_python_with_version(), Option, String (+2 more)

### Community 154 - "synapseGraphContext.ts"
Cohesion: 0.64
Nodes (9): buildAttentionShapeContext(), buildConvDenseShapeContext(), buildConvShapeContext(), buildDenseShapeContext(), buildNormShapeContext(), buildSynapseGraphContext(), nodeParams(), nodeType() (+1 more)

### Community 155 - "canvasModelBridge.ts"
Cohesion: 0.31
Nodes (4): CanvasModelMetadata, CanvasGraphIR, LayerConfig, ModelGraphConfig

### Community 156 - "parse_ir"
Cohesion: 0.43
Nodes (4): IRDataSpec, is_non_empty_ir(), parse_ir(), Any

### Community 157 - "layer_factory.py"
Cohesion: 0.46
Nodes (7): _bool(), create_layer(), _float(), _int(), Any, Module, Create nn.Module instances from IR node types.

### Community 158 - "shape_propagate.py"
Cohesion: 0.32
Nodes (6): _compatible(), CanvasGraphIR, Exception, Backend shape validation for Canvas IR., ShapeValidationError, validate_ir_shapes()

### Community 159 - "devScriptStorage.ts"
Cohesion: 0.50
Nodes (7): loadAllSavedScripts(), migrateLegacyDevScripts(), readKey(), readUserDevScripts(), StoredDevScript, testScriptsKey(), trainScriptsKey()

### Community 160 - "NotificationContext.tsx"
Cohesion: 0.18
Nodes (10): Notification, NotificationAction, NotificationContainerProps, NotificationContext, NotificationContextType, NotificationItem(), NotificationItemProps, NotificationOptions (+2 more)

### Community 161 - "GradientChatInput.tsx"
Cohesion: 0.60
Nodes (4): darkenHex(), GradientChatInput, GradientChatInputProps, hexLuminance()

### Community 162 - "Identifier"
Cohesion: 0.67
Nodes (3): Identifier, description, oneOf

### Community 163 - "Target"
Cohesion: 0.67
Nodes (3): Target, description, oneOf

### Community 164 - "DevTestPanel"
Cohesion: 0.50
Nodes (8): calculateAffectedLines(), deleteScript(), DevTestPanel(), getScriptsKey(), loadScripts(), saveScript(), ScriptLibraryModal(), updateScript()

### Community 165 - "CanvasInferenceTab.tsx"
Cohesion: 0.38
Nodes (6): CanvasInferenceResult, CanvasInferenceTab(), CanvasModelInfo, parseInputString(), pct(), Props

### Community 166 - "PageContext.tsx"
Cohesion: 0.47
Nodes (5): PageId, PageContext, PageContextProvider(), PageContextType, setCurrentScreen()

### Community 167 - "DatasetInfo"
Cohesion: 0.67
Nodes (4): DevTestPanelProps, DevTrainPanelProps, DatasetInfo, ModelInfo

## Ambiguous Edges - Review These
- `SynapseBuilder()` → `applyAutoFix()`  [AMBIGUOUS]
  src/components/synapse/ai/INTEGRATION_GUIDE.md · relation: calls

## Knowledge Gaps
- **679 isolated node(s):** `name`, `private`, `license`, `version`, `type` (+674 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `SynapseBuilder()` and `applyAutoFix()`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `AppState` connect `dataset_manager.rs` to `training_manager.rs`, `Database`, `analysis_manager.rs`, `String`, `model_manager.rs`, `main.rs`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `useLanguage()` connect `useLanguage` to `TrainingPanel.tsx`, `AppContextMenu.tsx`, `DevTestPanel.tsx`, `DiffViewer.tsx`, `LanguageContext.tsx`, `AnalysisPanel.tsx`, `TrainingDashboard.tsx`, `LaboratoryPanel.tsx`, `DevTrainPanel.tsx`, `plugins/datasetCompat.ts`, `SynapseBuilder.tsx`, `FloatingAICoach.tsx`, `ModelLibrary.tsx`, `AISettingsContext.tsx`, `Settings.tsx`, `DevTestPanel`, `OpenLibraryModal.tsx`, `CanvasInferenceTab.tsx`, `SynapseAIPanel.tsx`, `nodeTypes.ts`, `graphIR.ts`, `codeEdits.ts`, `SynapseAICoachPanel.tsx`, `DatasetUpload.tsx`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Why does `load_canvas_model()` connect `load_canvas_model` to `parse_ir`, `ir.py`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **What connects `name`, `private`, `license` to the rest of the system?**
  _679 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `dataset_manager.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.06993006993006994 - nodes in this community are weakly interconnected._
- **Should `training_manager.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.06287363430220573 - nodes in this community are weakly interconnected._