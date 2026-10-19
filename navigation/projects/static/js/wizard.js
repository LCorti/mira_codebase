/**
 * Setup Wizard JavaScript
 * Handles all the wizard navigation, validation, and interactions
 */

let currentStep = 1;
const totalSteps = 5;

const stepDescriptions = {
    1: "Provide some metadata to get started",
    2: "Now select a model",
    3: "The datasets and metrics next",
    4: "Which diagnostic tests would you like to run?",
    5: "Review the project parameters and configuration",
};

const stepButtonTexts = {
    1: { next: "Next: Model Selection", back: "Quit" },
    2: { next: "Next: Evaluation", back: "Back" },
    3: { next: "Next: Diagnostics", back: "Back" },
    4: { next: "Next: Review", back: "Back" },
    5: { next: "Create Project", back: "Back" },
};

/**
 * Updates the step indicator UI
 */
function updateStepIndicator() {
    // Update step counter
    document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
        const stepNumber = index + 1;

        // Remove all possible background classes
        indicator.classList.remove('bg-gray-300', 'bg-gray-500', 'bg-primary');

        if (currentStep === totalSteps) {
            // Final step - all indicators are primary
            indicator.classList.add('bg-primary');
        } else if (stepNumber < currentStep) {
            // Old/completed steps - gray-500
            indicator.classList.add('bg-gray-500');
        } else if (stepNumber === currentStep) {
            // Current step - primary
            indicator.classList.add('bg-primary');
        } else {
            // Future steps - gray-300
            indicator.classList.add('bg-gray-300');
        }
    });

    // Update description
    document.getElementById('step-description').textContent = stepDescriptions[currentStep];

    // Update button texts
    const backBtn = document.getElementById('back-btn-text');
    const nextBtn = document.getElementById('next-btn-text');
    backBtn.textContent = stepButtonTexts[currentStep].back;
    nextBtn.textContent = stepButtonTexts[currentStep].next;
}

/**
 * Shows a specific step and hides all others
 */
function showStep(step) {
    // Hide all steps
    document.querySelectorAll('.wizard-step').forEach(stepDiv => {
        stepDiv.classList.add('hidden');
    });

    // Show current step
    document.getElementById(`step-${step}`).classList.remove('hidden');

    // Update step indicator
    updateStepIndicator();

    // Scroll to top of page
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Filter diagnostic tests by task type if on step 4
    if (step === 4) {
        filterDiagnosticTestsByTask();
    }

    // Update review content if on step 5 (final step)
    if (step === totalSteps) {
        updateReviewContent();
    }
}

/**
 * Advances to the next step
 */
function nextStep() {
    if (currentStep === totalSteps) {
        // Before submitting, combine the two test radio selections into test_ids inputs
        const shouldKnowTest = document.querySelector('input[name="should_know_test"]:checked');
        const reallyKnowTest = document.querySelector('input[name="really_know_test"]:checked');

        // Remove any existing test_ids hidden inputs
        const existingTestIds = document.querySelectorAll('input[name="test_ids"]');
        existingTestIds.forEach(input => input.remove());

        // Add hidden inputs for each selected test
        const form = document.getElementById('setup-wizard-form');
        if (shouldKnowTest) {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'test_ids';
            input.value = shouldKnowTest.value;
            form.appendChild(input);
        }

        if (reallyKnowTest) {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'test_ids';
            input.value = reallyKnowTest.value;
            form.appendChild(input);
        }

        // Submit form on final step
        form.submit();
        return;
    }

    // Validate current step before proceeding
    if (validateCurrentStep()) {
        currentStep++;
        showStep(currentStep);
    }
}

/**
 * Goes back to the previous step
 */
function previousStep() {
    if (currentStep === 1) {
        // Quit - redirect to projects page
        window.location.href = projectsUrl; // This will be set in the main template
        return;
    }

    currentStep--;
    showStep(currentStep);
}

/**
 * Validates the current step before allowing navigation
 */
function validateCurrentStep() {
    switch (currentStep) {
        case 1:
            // Validate project name
            const projectName = document.querySelector('input[name="name"]');
            if (!projectName || !projectName.value.trim()) {
                alert('Please enter a project name to continue.');
                if (projectName) projectName.focus();
                return false;
            } else if (projectName.value.trim().length < 3) {
                alert('Project name must be at least 3 characters long.');
                projectName.focus();
                return false;
            }

            break;
        case 2:
            const selectedModel = document.querySelector('input[name="selected_model"]:checked');
            const customModel = document.getElementById('custom_model').value.trim();

            if (!selectedModel && !customModel) {
                alert('Please select a model or load a custom model to continue.');
                return false;
            }

            // Validate model parameters
            const modelParametersTextarea = document.getElementById('model_parameters');
            if (modelParametersTextarea && modelParametersTextarea.value.trim()) {
                try {
                    const modelParams = JSON.parse(modelParametersTextarea.value);
                    if (!modelParams.model || !modelParams.tokenizer) {
                        alert('Model parameters must contain both "model" and "tokenizer" sections.');
                        return false;
                    }
                    if (!modelParams.model.from_pretrained || !modelParams.model.inference) {
                        alert('Model section must contain "from_pretrained" and "inference" subsections.');
                        return false;
                    }
                    if (!modelParams.tokenizer.from_pretrained || !modelParams.tokenizer.inference) {
                        alert('Tokenizer section must contain "from_pretrained" and "inference" subsections.');
                        return false;
                    }
                } catch (e) {
                    alert('Please fix the JSON syntax errors in the model parameters before continuing.');
                    return false;
                }
            }
            break;
        case 3:
            const taskType = document.getElementById('task_type');
            if (!taskType || !taskType.value.trim()) {
                alert('Please select a task type to continue.');
                if (taskType) taskType.focus();
                return false;
            }

            // Validate task parameters
            const taskTypeValue = taskType.value.trim();
            if (taskTypeValue === 'text-retrieval') {
                if (!validateRAGTaskParameters()) {
                    return false;
                }
            } else if (taskTypeValue === 'summarization') {
                if (!validateSummarizationTaskParameters()) {
                    return false;
                }
            } else if (taskTypeValue == 'text-simplification') {
                if (!validateTextSimplificationTaskParameters()) {
                    return false;
                }
            } else {
                alert('Please select a valid task type to continue.');
                if (taskType) taskType.focus();
                return false;
            }

            const selectedDatasets = document.querySelectorAll('input[name="datasets"]:checked');

            if (selectedDatasets.length === 0) {
                alert('Please select at least one dataset to continue.');
                return false;
            }

            // Check that each selected dataset has at least one metric selected
            let hasMetrics = false;
            const associations = {};

            selectedDatasets.forEach(datasetCheckbox => {
                const datasetValue = datasetCheckbox.value;
                const datasetMetrics = document.querySelectorAll(`input[name="dataset_metrics"][data-dataset="${datasetValue}"]:checked`);
                associations[datasetValue] = Array.from(datasetMetrics).map(m => m.value);
                if (datasetMetrics.length > 0) {
                    hasMetrics = true;
                }
            });

            if (!hasMetrics) {
                alert('Please select at least one metric for your selected datasets.');
                return false;
            }

            // Validate dataset parameters if they exist
            const datasetParametersTextarea = document.getElementById('dataset_parameters');
            if (datasetParametersTextarea && datasetParametersTextarea.value.trim()) {
                try {
                    const datasetParams = JSON.parse(datasetParametersTextarea.value);

                    // Get current task type to determine which fields are required
                    const currentTaskType = taskType.value.trim();
                    const needsReferenceField = currentTaskType === 'summarization' || currentTaskType === 'text-simplification';
                    const needsQueryIdField = currentTaskType === 'text-retrieval';

                    // Check that all selected datasets have parameters
                    for (const dataset of selectedDatasets) {
                        const datasetName = dataset.value;
                        if (!datasetParams[datasetName]) {
                            alert(`Dataset parameters missing for "${datasetName}". Please configure dataset parameters.`);
                            return false;
                        }

                        const params = datasetParams[datasetName];

                        // Validate required fields for all tasks
                        if (!params.split || !params.input_column) {
                            alert(`Incomplete parameters for dataset "${datasetName}". Please ensure split and input_column are filled.`);
                            return false;
                        }

                        // Conditionally validate task-specific fields
                        if (needsReferenceField && !params.reference_column) {
                            alert(`Dataset "${datasetName}" is missing reference_column (required for ${currentTaskType}).`);
                            return false;
                        }

                        if (needsQueryIdField && !params.query_id_column) {
                            alert(`Dataset "${datasetName}" is missing query_id_column (required for ${currentTaskType}).`);
                            return false;
                        }

                        // Validate num_samples if present
                        if (params.num_samples !== undefined && (typeof params.num_samples !== 'number' || params.num_samples < 1)) {
                            alert(`Invalid num_samples for dataset "${datasetName}". Must be a positive number.`);
                            return false;
                        }
                    }
                } catch (e) {
                    alert('Dataset parameters contain invalid JSON. Please check the configuration.');
                    return false;
                }
            }

            // Update the hidden field with dataset-metric associations
            updateDatasetMetricAssociations();
            break;
        case 4:
            // Validate that one test is selected from each category
            const shouldKnowTest = document.querySelector('input[name="should_know_test"]:checked');
            const reallyKnowTest = document.querySelector('input[name="really_know_test"]:checked');

            if (!shouldKnowTest) {
                alert('Please select one test from the Should-Know category.');
                return false;
            }

            if (!reallyKnowTest) {
                alert('Please select one test from the Really-Know category.');
                return false;
            }
            break;
        case 5:
            // Validate JSON parameters
            const parametersTextarea = document.getElementById('parameters');
            if (!parametersTextarea || !parametersTextarea.value.trim()) {
                alert('Please configure the project parameters.');
                if (parametersTextarea) parametersTextarea.focus();
                return false;
            }

            try {
                JSON.parse(parametersTextarea.value);
            } catch (e) {
                alert('Please fix the JSON syntax errors in the parameters before continuing.');
                if (parametersTextarea) parametersTextarea.focus();
                return false;
            }

            // Auto-generate parameters if needed
            if (parametersTextarea.value.trim() === '' || parametersTextarea.value.includes('DATASET_NAME')) {
                generateParameters();
            }
            break;
    }
    return true;
}

/**
 * Updates the review content on the final step
 */
function updateReviewContent() {
    const reviewContainer = document.getElementById('review-items-container');
    let reviewItems = [];

    // Project Name
    const projectName = document.querySelector('input[name="name"]');
    const projectDescription = document.querySelector('textarea[name="project_description"]');
    if (projectName && projectName.value) {
        reviewItems.push({
            id: 'project-info',
            icon: 'fa-solid fa-folder-open',
            iconColor: 'text-blue-500',
            title: 'Project Information',
            subtitle: projectName.value,
            details: projectDescription && projectDescription.value ? projectDescription.value : null,
            hasJson: false
        });
    }

    // Model selection - check both regular and custom models
    const selectedModel = document.querySelector('input[name="selected_model"]:checked');
    const customModel = document.getElementById('custom_model').value.trim();

    let modelName = null;
    if (customModel) {
        modelName = `${customModel} (Custom)`;
    } else if (selectedModel) {
        const modelLabel = selectedModel.closest('label').querySelector('h3').textContent;
        modelName = modelLabel;
    }

    if (modelName) {
        const modelParametersTextarea = document.getElementById('model_parameters');
        let modelParams = null;
        if (modelParametersTextarea && modelParametersTextarea.value.trim()) {
            try {
                modelParams = JSON.parse(modelParametersTextarea.value);
            } catch (e) {
                // Invalid JSON
            }
        }

        reviewItems.push({
            id: 'model-config',
            icon: 'fa-solid fa-brain',
            iconColor: 'text-purple-500',
            title: 'Model Configuration',
            subtitle: modelName,
            details: modelParams ? 'Custom parameters configured' : 'Default parameters',
            hasJson: !!modelParams,
            jsonData: modelParams
        });
    }

    // Task Configuration
    const taskParametersTextarea = document.getElementById('task_parameters');
    let taskParams = null;
    if (taskParametersTextarea && taskParametersTextarea.value.trim()) {
        try {
            taskParams = JSON.parse(taskParametersTextarea.value);
        } catch (e) {
            // Invalid JSON
        }
    }

    if (taskParams && taskParams.task_type) {
        const taskTypeDisplayNames = {
            'text-retrieval': 'Retrieval-Augmented Generation',
            'summarization': 'Text Summarization',
            'simplification': 'Text Simplification'
        };

        const taskDisplayName = taskTypeDisplayNames[taskParams.task_type] || taskParams.task_type;
        const paramCount = Object.keys(taskParams.task_parameters || {}).length;

        reviewItems.push({
            id: 'task-config',
            icon: 'fa-solid fa-cog',
            iconColor: 'text-indigo-500',
            title: 'Task Configuration',
            subtitle: taskDisplayName,
            details: `${paramCount} custom parameters configured`,
            hasJson: true,
            jsonData: taskParams
        });
    }

    // Datasets and their parameters
    const selectedDatasets = Array.from(document.querySelectorAll('input[name="datasets"]:checked'));
    if (selectedDatasets.length > 0) {
        let datasetDetails = [];
        selectedDatasets.forEach(datasetCheckbox => {
            const datasetValue = datasetCheckbox.value;
            const datasetLabel = datasetCheckbox.closest('label').querySelector('h4').textContent;
            datasetDetails.push(datasetLabel);
        });

        const datasetParametersTextarea = document.getElementById('dataset_parameters');
        let datasetParams = null;
        if (datasetParametersTextarea && datasetParametersTextarea.value.trim()) {
            try {
                datasetParams = JSON.parse(datasetParametersTextarea.value);
            } catch (e) {
                // Invalid JSON
            }
        }

        reviewItems.push({
            id: 'dataset-config',
            icon: 'fa-solid fa-database',
            iconColor: 'text-green-500',
            title: 'Dataset Configuration',
            subtitle: `${selectedDatasets.length} dataset(s) selected`,
            details: datasetDetails.join(', '),
            hasJson: !!datasetParams,
            jsonData: datasetParams
        });
    }

    // Metrics per dataset
    const selectedDatasetMetrics = Array.from(document.querySelectorAll('input[name="dataset_metrics"]:checked'));
    if (selectedDatasetMetrics.length > 0) {
        let metricsByDataset = {};
        selectedDatasetMetrics.forEach(metricCheckbox => {
            const dataset = metricCheckbox.dataset.dataset;
            const metricName = metricCheckbox.value;
            if (!metricsByDataset[dataset]) {
                metricsByDataset[dataset] = [];
            }
            metricsByDataset[dataset].push(metricName);
        });

        let metricDetails = [];
        Object.entries(metricsByDataset).forEach(([dataset, metrics]) => {
            const datasetCheckbox = document.querySelector(`input[name="datasets"][value="${dataset}"]`);
            const datasetLabel = datasetCheckbox ? datasetCheckbox.closest('label').querySelector('h4').textContent : dataset;
            metricDetails.push(`${datasetLabel}: ${metrics.join(', ')}`);
        });

        reviewItems.push({
            id: 'metrics-config',
            icon: 'fa-solid fa-chart-line',
            iconColor: 'text-orange-500',
            title: 'Metrics Configuration',
            subtitle: `${Object.keys(metricsByDataset).length} dataset(s) configured`,
            details: metricDetails.join(' | '),
            hasJson: true,
            jsonData: metricsByDataset
        });
    }

    // Selected diagnostic tests
    const shouldKnowTest = document.querySelector('input[name="should_know_test"]:checked');
    const reallyKnowTest = document.querySelector('input[name="really_know_test"]:checked');

    if (shouldKnowTest || reallyKnowTest) {
        const testNames = [];

        if (shouldKnowTest) {
            const shouldKnowLabel = shouldKnowTest.closest('label').querySelector('h3').textContent;
            testNames.push(`Should-Know: ${shouldKnowLabel}`);
        }

        if (reallyKnowTest) {
            const reallyKnowLabel = reallyKnowTest.closest('label').querySelector('h3').textContent;
            testNames.push(`Really-Know: ${reallyKnowLabel}`);
        }

        reviewItems.push({
            id: 'tests-config',
            icon: 'fa-solid fa-flask',
            iconColor: 'text-red-500',
            title: 'Diagnostic Tests',
            subtitle: `${testNames.length} test(s) selected`,
            details: testNames.join(', '),
            hasJson: false
        });
    }

    // Auto-run setting
    const autoRun = document.querySelector('input[name="auto_run"]');
    if (autoRun && autoRun.checked) {
        reviewItems.push({
            id: 'autorun-config',
            icon: 'fa-solid fa-play',
            iconColor: 'text-cyan-500',
            title: 'Auto-run Configuration',
            subtitle: 'Enabled',
            details: 'Project will run automatically after creation',
            hasJson: false
        });

        // Compute how many methods will be queued based on selected tests and datasets
        try {
            const autoRunWarningEl = document.getElementById('auto-run-warning');
            const autoRunWarningText = document.getElementById('auto-run-warning-detail');

            // Determine selected datasets count
            const selectedDatasetsForCount = Array.from(document.querySelectorAll('input[name="datasets"]:checked'));
            const datasetCount = selectedDatasetsForCount.length;

            // Sum methods from selected tests
            let methodsPerTestTotal = 0;
            const selectedTestsForCount = [];
            if (shouldKnowTest) selectedTestsForCount.push(shouldKnowTest);
            if (reallyKnowTest) selectedTestsForCount.push(reallyKnowTest);

            selectedTestsForCount.forEach(testInput => {
                try {
                    const trigger = testInput.closest('label').querySelector('.test-metadata-trigger');
                    if (trigger) {
                        const infoStr = trigger.getAttribute('data-tooltip-info');
                        if (infoStr) {
                            const parsed = JSON.parse(infoStr);
                            if (Array.isArray(parsed)) {
                                methodsPerTestTotal += parsed.length;
                            }
                        }
                    }
                } catch (e) {
                    // ignore parsing errors and assume 0 for that test
                }
            });

            // Number of queued jobs is determined by the total number of methods
            // defined by the selected tests (not multiplied by dataset count).
            // Each project run will queue the methods defined by the tests.
            const queuedJobs = methodsPerTestTotal;

            if (autoRunWarningEl && autoRunWarningText) {
                if (queuedJobs > 0 && datasetCount > 0) {
                    autoRunWarningText.innerHTML = `Based on your selected tests, <strong>${queuedJobs}</strong> job(s) will be queued. This will take some time depending on the model and datasets selected. Are you sure you want to proceed with auto-running the project?`;
                } else if (queuedJobs > 0 && datasetCount === 0) {
                    // Methods present but no dataset selected yet
                    autoRunWarningText.innerHTML = `You have <strong>${queuedJobs}</strong> job(s) from the selected tests, but no datasets are selected yet. Select at least one dataset to queue jobs.`;
                } else {
                    autoRunWarningText.innerHTML = `No jobs detected for the chosen tests; <strong>0</strong> jobs will be queued.`;
                }
                autoRunWarningEl.classList.remove('hidden');
            }
        } catch (e) {
            // fail silently if anything goes wrong computing the warning
            console.warn('Failed to compute auto-run warning:', e);
        }
    } else {
        // Hide the auto-run warning if auto-run is not selected
        const autoRunWarningEl = document.getElementById('auto-run-warning');
        if (autoRunWarningEl) autoRunWarningEl.classList.add('hidden');
    }

    // Generate HTML for review items
    if (reviewItems.length === 0) {
        reviewContainer.innerHTML = `
            <div class="px-6 py-8 text-center text-gray-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4 text-gray-300"></i>
                <p>No configuration selected</p>
                <p class="text-sm">Please go back and configure your project.</p>
            </div>
        `;
        return;
    }

    let content = '';
    reviewItems.forEach((item, index) => {
        content += `
            <div class="review-item">
                <div class="px-6 py-4 cursor-pointer hover:bg-gray-50 transition-colors duration-200 flex items-center justify-between" 
                    ${item.hasJson ? `onclick="toggleReviewSection('${item.id}')"` : ''}>
                    <div class="flex items-center">
                        <i class="${item.icon} ${item.iconColor} mr-3 text-lg"></i>
                        <div>
                            <h3 class="font-medium text-gray-800">${item.title}</h3>
                            <p class="text-sm text-gray-600">${item.subtitle}</p>
                            ${item.details ? `<p class="text-xs text-gray-500 mt-1">${item.details}</p>` : ''}
                        </div>
                    </div>
                    ${item.hasJson ? `
                        <div class="flex items-center text-gray-400">
                            <span class="text-xs mr-2">View JSON</span>
                            <i id="${item.id}-chevron" class="fa-solid fa-chevron-right transition-transform duration-200"></i>
                        </div>
                    ` : ''}
                </div>
                ${item.hasJson ? `
                    <div id="${item.id}-content" class="hidden px-6 py-4 bg-gray-50 border-t border-gray-200">
                        <div class="mb-2">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Configuration Parameters (Read-only)</label>
                        </div>
                        <textarea 
                            readonly
                            rows="8"
                            class="w-full px-3 py-2 font-mono text-xs bg-white border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        >${JSON.stringify(item.jsonData, null, 2)}</textarea>
                    </div>
                ` : ''}
            </div>
        `;
    });

    reviewContainer.innerHTML = content;
}

/**
 * Toggles a review section's JSON content
 */
function toggleReviewSection(sectionId) {
    const content = document.getElementById(`${sectionId}-content`);
    const chevron = document.getElementById(`${sectionId}-chevron`);

    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        chevron.classList.remove('fa-chevron-right');
        chevron.classList.add('fa-chevron-down');
    } else {
        content.classList.add('hidden');
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-right');
    }
}

/**
 * Toggles the auto-run setting from the review screen
 */
function toggleAutoRun() {
    const autoRunCheckbox = document.querySelector('input[name="auto_run"]');
    if (autoRunCheckbox) {
        autoRunCheckbox.checked = false;
        // Update the review content to reflect the change
        updateReviewContent();
    }
}

/**
 * Updates the visual selection indicator for models
 */
function updateSelectionIndicator(radioInput) {
    // Reset all selection bars to gray
    document.querySelectorAll('.model-selection-bar').forEach(bar => {
        bar.classList.remove('bg-primary');
        bar.classList.add('bg-gray-300');
    });

    // Update the selected item's bar to primary color
    if (radioInput.checked) {
        const selectionBar = radioInput.closest('label').querySelector('.model-selection-bar');
        selectionBar.classList.remove('bg-gray-300');
        selectionBar.classList.add('bg-primary');

        // Clear custom model input when a regular model is selected
        document.getElementById('custom_model').value = '';

        // Remove custom model container border
        const customModelContainer = document.getElementById('custom-model-container');
        customModelContainer.classList.remove('border-primary');
        customModelContainer.classList.add('border-gray-300');

        // Auto-populate model parameters based on selected model
        populateModelParameters(radioInput.value);
    }
}

/**
 * Toggles the available models section
 */
function toggleAvailableModels() {
    toggleModelCard('available-models');
}

/**
 * Toggles the custom model input section
 */
function toggleCustomModel() {
    toggleModelCard('custom-model');
}

/**
 * Toggles the model parameters section
 */
function toggleModelParameters() {
    toggleModelCard('model-parameters');
}

/**
 * Generic function to toggle model cards (independent behavior)
 */
function toggleModelCard(cardName) {
    const content = document.getElementById(`${cardName}-content`);
    const chevron = document.getElementById(`${cardName}-chevron`);

    // Toggle this card's visibility
    if (content.classList.contains('hidden')) {
        // Open this card
        content.classList.remove('hidden');
        chevron.classList.remove('fa-chevron-right');
        chevron.classList.add('fa-chevron-down');
    } else {
        // Close this card
        content.classList.add('hidden');
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-right');
    }
}

/**
 * Loads a custom model from HuggingFace
 */
function loadCustomModel() {
    const modelId = document.getElementById('custom_model').value.trim();

    if (!modelId) {
        alert('Please enter a HuggingFace model ID');
        return;
    }

    // Basic validation for HuggingFace model ID format
    if (!modelId.includes('/') || modelId.split('/').length !== 2) {
        alert('Please enter a valid HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)');
        return;
    }

    // Check if the model ID already exists in the list
    const existingModels = Array.from(document.querySelectorAll('input[name="selected_model"]'))
        .map(input => input.value);
    if (existingModels.includes(modelId)) {
        alert('This model is already downloaded.');
        return;
    }

    // Show loading state
    const loadButton = event.target;
    const originalText = loadButton.innerHTML;
    loadButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-sm"></i><span>Loading...</span>';
    loadButton.disabled = true;

    // Make request to get model size
    const url = `${apiDownloadSizeUrl}?model_name=${encodeURIComponent(modelId)}`;
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Convert bytes to MB (1 MB = 1,048,576 bytes)
            const sizeInMB = (data.size / (1024 * 1024)).toFixed(2);

            // Show modal with model information
            showModelConfirmationModal(modelId, `${sizeInMB} MB`);
        })
        .catch(error => {
            console.error('Error fetching model size:', error);
            alert(`Error getting model size.`);
        })
        .finally(() => {
            // Restore button state
            loadButton.innerHTML = originalText;
            loadButton.disabled = false;
        });
}

/**
 * Shows the model confirmation modal
 */
function showModelConfirmationModal(modelName, size) {
    document.getElementById('modal-model-name').textContent = modelName;
    document.getElementById('modal-model-size').textContent = size;
    document.getElementById('model-size-modal').classList.remove('hidden');
}

/**
 * Closes the model confirmation modal
 */
function closeModal() {
    document.getElementById('model-size-modal').classList.add('hidden');
    document.getElementById('custom_model').value = ''; // Clear input field
    document.getElementById('modal-model-name').textContent = '';
    document.getElementById('modal-model-size').textContent = '';
    const loadButton = document.querySelector('button[onclick="loadCustomModel()"]');
    if (loadButton) {
        loadButton.innerHTML = '<i class="fa-solid fa-download text-sm"></i><span>Load</span>';
        loadButton.disabled = false;
    }
}

/**
 * Confirms the model download
 */
function confirmDownload() {
    const modelName = document.getElementById('modal-model-name').textContent;
    const modelSize = document.getElementById('modal-model-size').textContent;

    // Clear any selected model radios before adding the new one
    document.querySelectorAll('input[name="selected_model"]').forEach(radio => {
        radio.checked = false;
    });

    // Reset all selection bars to gray
    document.querySelectorAll('.model-selection-bar').forEach(bar => {
        bar.classList.remove('bg-primary');
        bar.classList.add('bg-gray-300');
    });

    document.getElementById('model-size-modal').classList.add('hidden');

    // Add the custom model to the available models list
    const config = addCustomModelToList(modelName, modelSize);

    // Remove custom model container border since model is now in the list
    document.getElementById('custom-model-container').classList.remove('border-primary');
    document.getElementById('custom-model-container').classList.add('border-gray-300');

    // Clear the custom model input
    document.getElementById('custom_model').value = '';

    // Update load button to show success
    const loadButton = document.getElementById('load-custom-model-btn');
    const originalText = loadButton.innerHTML;

    loadButton.innerHTML = '<i class="fa-solid fa-check text-sm"></i><span>Success</span>';
    loadButton.classList.remove('bg-gray-500', 'hover:bg-gray-600');
    loadButton.classList.add('bg-green-500');
    loadButton.disabled = true;

    setTimeout(() => {
        loadButton.innerHTML = originalText;
        loadButton.disabled = false;
        loadButton.classList.remove('bg-green-500');
        loadButton.classList.add('bg-gray-500', 'hover:bg-gray-600');
    }, 3000);

    // Auto-populate model parameters for custom model
    populateCustomModelParameters(modelName, config);
}

/**
 * Adds a custom model to the available models list
 */
function addCustomModelToList(modelName, modelSize) {
    // Find the models container
    const modelsContainer = document.querySelector('#available-models-content .overflow-y-auto');
    const emptyMessage = modelsContainer.querySelector('.text-center');

    // Remove empty message if it exists
    if (emptyMessage && emptyMessage.textContent.includes('No models found')) {
        emptyMessage.remove();
    }

    // Create new model item
    const modelItem = document.createElement('label');
    modelItem.className = 'project-item group px-6 py-4 hover:bg-gray-50 transition-colors duration-200 flex items-center justify-between cursor-pointer relative';

    // Initialize variables for model config data
    let config = {};
    let architecture = "Unknown";
    let contextLength = "Unknown";

    // Create the initial HTML with placeholder values
    modelItem.innerHTML = `
        <!-- Selection Indicator Bar -->
        <div class="absolute left-0 top-0 bottom-0 w-1 bg-primary transition-colors duration-200 model-selection-bar"></div>
        
        <!-- Hidden Radio Input -->
        <input type="radio" name="selected_model" value="${modelName}" class="hidden model-radio" onchange="updateSelectionIndicator(this)" checked>
        
        <div>
            <div class="flex items-center">
                <i class="fa-solid fa-brain text-purple-500 mr-3 ml-2"></i>
                <div>
                    <div class="flex items-center gap-2">
                        <h3 class="font-medium text-gray-800">${modelName}</h3>
                    </div>
                    <p class="text-sm text-gray-500">Custom HuggingFace model</p>
                </div>
            </div>
        </div>
        <div class="flex items-center relative">
            <span class="text-xs bg-primary text-white px-2 py-1 rounded mr-4">Not Downloaded Yet</span>
            <i class="fa-solid fa-circle-info text-gray-400 hover:text-primary cursor-pointer transition-colors duration-200 tooltip-trigger" 
                data-tooltip-title="Model Information"
                data-tooltip-info='[{"label": "Size", "value": "${modelSize}"}, {"label": "Context Length", "value": "${contextLength}"}, {"label": "Architecture", "value": "${architecture}"}]'></i>
        </div>
    `;

    // Insert at the top of the models list
    modelsContainer.insertBefore(modelItem, modelsContainer.firstChild);

    // Make request to get model config and update the tooltip
    const configUrl = `${apiModelConfigUrl}?model_name=${encodeURIComponent(modelName)}`;
    fetch(configUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.config) {
                try {
                    config = JSON.parse(data.config);
                } catch (e) {
                    console.warn('Error parsing config JSON:', e);
                    config = data.config; // Use as-is if it's already an object
                }
            }

            // Extract values from config
            contextLength = config.max_position_embeddings || "Unknown";
            architecture = (config.architectures && config.architectures[0]) || "Unknown";

            // Update the tooltip with the new information
            const tooltipElement = modelItem.querySelector('.tooltip-trigger');
            if (tooltipElement) {
                const newTooltipInfo = JSON.stringify([
                    { "label": "Size", "value": modelSize },
                    { "label": "Context Length", "value": contextLength },
                    { "label": "Architecture", "value": architecture }
                ]);
                tooltipElement.setAttribute('data-tooltip-info', newTooltipInfo);
            }
        })
        .catch(error => {
            console.error('Error fetching model config:', error);
            alert(`Error fetching model config.`);
        });

    // Automatically select this model
    const radioInput = modelItem.querySelector('input[name="selected_model"]');
    radioInput.checked = true;

    // Update selection indicator
    updateSelectionIndicator(radioInput);

    return config;
}

/**
 * Populates model parameters based on selected model configuration
 */
function populateModelParameters(modelName) {
    // Get model data from the models variable passed from Django
    if (typeof modelsData !== 'undefined' && modelsData[modelName]) {
        const modelConfig = modelsData[modelName];
        generateModelParametersFromConfig(modelConfig);
    } else {
        // Fallback to default parameters if model data not available
        generateDefaultModelParameters();
    }
}

/**
 * Populates model parameters for custom models using API
 */
function populateCustomModelParameters(modelName, config = {}) {
    if (config && Object.keys(config).length > 0) {
        generateModelParametersFromConfig(config);
        return;
    }

    // Check if API config URL is available for fetching config
    if (typeof apiModelConfigUrl !== 'undefined') {
        const url = `${apiModelConfigUrl}?model_name=${encodeURIComponent(modelName)}`;
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                let modelConfig = {};
                if (data.config) {
                    try {
                        modelConfig = JSON.parse(data.config);
                    } catch (e) {
                        console.warn('Error parsing config JSON:', e);
                        modelConfig = data.config; // Use as-is if it's already an object
                    }
                }
                generateModelParametersFromConfig(modelConfig);
            })
            .catch(error => {
                console.warn('Error fetching model config, using defaults:', error);
                generateDefaultModelParameters();
            });
    } else {
        console.warn('API config URL not available, using default parameters');
        generateDefaultModelParameters();
    }
}

/**
 * Generates model parameters from model configuration
 */
function generateModelParametersFromConfig(modelConfig) {
    const parameters = {
        model: {
            from_pretrained: {},
            inference: {}
        },
        tokenizer: {
            from_pretrained: {},
            inference: {}
        }
    };

    // Model from_pretrained parameters
    parameters.model.from_pretrained = {
        trust_remote_code: false,
        torch_dtype: modelConfig.torch_dtype === 'float32' ? 'torch.float32' :
            modelConfig.torch_dtype === 'float16' ? 'torch.float16' :
                modelConfig.use_bfloat16 ? 'torch.bfloat16' : 'auto',
        device_map: 'auto',
        attn_implementation: 'eager'
    };

    // Model inference parameters from config
    parameters.model.inference = {
        max_new_tokens: modelConfig.max_new_tokens || 1024,
        min_length: modelConfig.min_length || 0,
        early_stopping: modelConfig.early_stopping || false,
        do_sample: modelConfig.do_sample || false,
        num_beams: modelConfig.num_beams || 1,
        temperature: modelConfig.temperature || 0.7,
        top_k: modelConfig.top_k || 50,
        top_p: modelConfig.top_p || 0.95,
        repetition_penalty: modelConfig.repetition_penalty || 1.0,
        length_penalty: modelConfig.length_penalty || 1.0,
    };

    // Tokenizer from_pretrained parameters
    parameters.tokenizer.from_pretrained = {
        trust_remote_code: false,
        use_fast: true,
    };

    // Tokenizer inference parameters
    parameters.tokenizer.inference = {
        padding: true,
        truncation: true,
        return_tensors: 'pt',
        add_special_tokens: true,
    };

    // Update the textarea
    updateModelParametersTextarea(parameters);
}

/**
 * Generates default model parameters when config is not available
 */
function generateDefaultModelParameters() {
    const parameters = {
        model: {
            from_pretrained: {
                trust_remote_code: false,
                torch_dtype: 'auto',
                device_map: 'auto',
                attn_implementation: 'eager'
            },
            inference: {
                max_length: 512,
                temperature: 1.0,
                top_p: 1.0,
                do_sample: false,
                num_beams: 1,
                early_stopping: false,
                repetition_penalty: 1.0,
                length_penalty: 1.0
            }
        },
        tokenizer: {
            from_pretrained: {
                trust_remote_code: false,
                use_fast: true,
                padding_side: 'right',
                model_max_length: 512
            },
            inference: {
                padding: true,
                truncation: true,
                return_tensors: 'pt',
                add_special_tokens: true,
                max_length: 512
            }
        }
    };

    updateModelParametersTextarea(parameters);
}

/**
 * Updates the model parameters textarea with formatted JSON
 */
function updateModelParametersTextarea(parameters) {
    const textarea = document.getElementById('model_parameters');
    if (textarea) {
        textarea.value = JSON.stringify(parameters, null, 2);
        validateModelParametersJSON();
    }
}

/**
 * Resets model parameters to auto-generated defaults
 */
function resetModelParameters(event) {
    if (event) event.stopPropagation(); // Prevent toggle when clicking reset

    // Check if a model is selected
    const selectedModel = document.querySelector('input[name="selected_model"]:checked');
    const customModel = document.getElementById('custom_model').value.trim();

    if (selectedModel) {
        populateModelParameters(selectedModel.value);
    } else if (customModel) {
        populateCustomModelParameters(customModel);
    } else {
        generateDefaultModelParameters();
    }
}

/**
 * Validates the JSON in the model parameters textarea
 */
function validateModelParametersJSON() {
    const textarea = document.getElementById('model_parameters');
    const statusElement = document.getElementById('model-parameters-status');
    const errorElement = document.getElementById('model-parameters-error');
    const errorTextElement = document.getElementById('model-parameters-error-text');

    if (!textarea) return;

    const value = textarea.value.trim();

    if (!value) {
        // Hide validation indicators if empty
        if (statusElement) statusElement.textContent = 'Empty';
        if (errorElement) errorElement.classList.add('hidden');
        return;
    }

    try {
        const parsed = JSON.parse(value);

        // Validate structure
        if (!parsed.model || !parsed.tokenizer) {
            throw new Error('Missing required "model" or "tokenizer" sections');
        }

        if (!parsed.model.from_pretrained || !parsed.model.inference) {
            throw new Error('Model section must contain "from_pretrained" and "inference" subsections');
        }

        if (!parsed.tokenizer.from_pretrained || !parsed.tokenizer.inference) {
            throw new Error('Tokenizer section must contain "from_pretrained" and "inference" subsections');
        }

        // Valid JSON and structure
        if (statusElement) {
            statusElement.textContent = 'Valid JSON';
            statusElement.className = 'text-xs px-2 py-1 rounded-full bg-green-100 text-green-800';
        }
        if (errorElement) errorElement.classList.add('hidden');

        textarea.classList.remove('border-red-300');
        textarea.classList.add('border-gray-300');

    } catch (e) {
        // Invalid JSON or structure
        if (statusElement) {
            statusElement.textContent = 'Invalid JSON';
            statusElement.className = 'text-xs px-2 py-1 rounded-full bg-red-100 text-red-800';
        }
        if (errorElement) {
            errorElement.classList.remove('hidden');
            if (errorTextElement) errorTextElement.textContent = e.message;
        }

        textarea.classList.remove('border-gray-300');
        textarea.classList.add('border-red-300');
    }
}

/**
 * Updates the visual selection indicator for datasets
 */
function updateDatasetSelection(checkboxInput) {
    const selectionBar = checkboxInput.closest('label').querySelector('.dataset-selection-bar');

    if (checkboxInput.checked) {
        selectionBar.classList.remove('bg-gray-300');
        selectionBar.classList.add('bg-primary');
    } else {
        selectionBar.classList.remove('bg-primary');
        selectionBar.classList.add('bg-gray-300');
    }

    // Update metrics display to show per-dataset selection
    updatePerDatasetMetrics();

    // Update dataset parameters forms
    updateDatasetParametersForms();
}

/**
 * Filters diagnostic tests based on the selected task type
 */
function filterDiagnosticTestsByTask() {
    const taskParametersTextarea = document.getElementById('task_parameters');
    let selectedTaskType = null;

    // Get the selected task type from task parameters
    if (taskParametersTextarea && taskParametersTextarea.value.trim()) {
        try {
            const taskConfig = JSON.parse(taskParametersTextarea.value);
            selectedTaskType = taskConfig.task_type;
        } catch (e) {
            // If parsing fails, try to get from task type selector
            const taskTypeSelector = document.querySelector('select[name="task_type"]');
            if (taskTypeSelector) {
                selectedTaskType = taskTypeSelector.value;
            }
        }
    }

    // If no task type is selected, show all tests
    if (!selectedTaskType) {
        document.querySelectorAll('.test-item').forEach(item => {
            item.style.display = '';
        });
        updateDiagnosticEmptyMessages();
        return;
    }

    // Filter Should-Know tests
    const shouldKnowItems = document.querySelectorAll('input[name="should_know_test"]');
    let shouldKnowVisible = 0;
    shouldKnowItems.forEach(input => {
        const testItem = input.closest('.test-item');
        const tasks = testItem.getAttribute('data-tasks');

        if (tasks && tasks.split(',').includes(selectedTaskType)) {
            testItem.style.display = '';
            shouldKnowVisible++;
        } else {
            testItem.style.display = 'none';
            // Uncheck if it was selected
            if (input.checked) {
                input.checked = false;
                const selectionBar = testItem.querySelector('.test-selection-bar');
                if (selectionBar) {
                    selectionBar.classList.remove('bg-blue-500');
                    selectionBar.classList.add('bg-gray-300');
                }
            }
        }
    });

    // Filter Really-Know tests
    const reallyKnowItems = document.querySelectorAll('input[name="really_know_test"]');
    let reallyKnowVisible = 0;
    reallyKnowItems.forEach(input => {
        const testItem = input.closest('.test-item');
        const tasks = testItem.getAttribute('data-tasks');

        if (tasks && tasks.split(',').includes(selectedTaskType)) {
            testItem.style.display = '';
            reallyKnowVisible++;
        } else {
            testItem.style.display = 'none';
            // Uncheck if it was selected
            if (input.checked) {
                input.checked = false;
                const selectionBar = testItem.querySelector('.test-selection-bar');
                if (selectionBar) {
                    selectionBar.classList.remove('bg-purple-500');
                    selectionBar.classList.add('bg-gray-300');
                }
            }
        }
    });

    // Update empty messages
    updateDiagnosticEmptyMessages(shouldKnowVisible, reallyKnowVisible, selectedTaskType);
}

/**
 * Updates the empty messages for diagnostic test sections
 */
function updateDiagnosticEmptyMessages(shouldKnowVisible, reallyKnowVisible, taskType) {
    // Update Should-Know empty message
    const shouldKnowContainer = document.querySelector('input[name="should_know_test"]')?.closest('.max-h-\\[400px\\]');
    if (shouldKnowContainer) {
        let emptyMessage = shouldKnowContainer.querySelector('.text-center.text-gray-500');

        if (shouldKnowVisible === 0) {
            if (!emptyMessage) {
                emptyMessage = document.createElement('div');
                emptyMessage.className = 'px-6 py-8 text-center text-gray-500';
                shouldKnowContainer.appendChild(emptyMessage);
            }
            const taskName = taskType ? formatTaskName(taskType) : 'this task';
            emptyMessage.innerHTML = `
                <i class="fas fa-flask text-4xl mb-4 text-gray-300"></i>
                <p>No Should-Know methods found for ${taskName}</p>
            `;
            emptyMessage.style.display = '';
        } else if (emptyMessage) {
            emptyMessage.style.display = 'none';
        }
    }

    // Update Really-Know empty message
    const reallyKnowContainer = document.querySelector('input[name="really_know_test"]')?.closest('.max-h-\\[400px\\]');
    if (reallyKnowContainer) {
        let emptyMessage = reallyKnowContainer.querySelector('.text-center.text-gray-500');

        if (reallyKnowVisible === 0) {
            if (!emptyMessage) {
                emptyMessage = document.createElement('div');
                emptyMessage.className = 'px-6 py-8 text-center text-gray-500';
                reallyKnowContainer.appendChild(emptyMessage);
            }
            const taskName = taskType ? formatTaskName(taskType) : 'this task';
            emptyMessage.innerHTML = `
                <i class="fas fa-flask text-4xl mb-4 text-gray-300"></i>
                <p>No Really-Know methods found for ${taskName}</p>
            `;
            emptyMessage.style.display = '';
        } else if (emptyMessage) {
            emptyMessage.style.display = 'none';
        }
    }
}

/**
 * Formats task type name for display
 */
function formatTaskName(taskType) {
    const taskNames = {
        'text-retrieval': 'Text Retrieval',
        'text-simplification': 'Text Simplification',
        'summarization': 'Summarization'
    };
    return taskNames[taskType] || taskType;
}

/**
 * Updates the visual selection indicator for tests
 */
function updateTestSelection(radioInput, category) {
    // Reset all selection bars in this category
    const categoryName = category === 'should-know' ? 'should_know_test' : 'really_know_test';
    const allRadios = document.querySelectorAll(`input[name="${categoryName}"]`);

    allRadios.forEach(radio => {
        const selectionBar = radio.closest('label').querySelector('.test-selection-bar');
        if (radio.checked) {
            selectionBar.classList.remove('bg-gray-300');
            selectionBar.classList.add('bg-primary');
        } else {
            selectionBar.classList.remove('bg-primary');
            selectionBar.classList.add('bg-gray-300');
        }
    });
}

/**
 * Generates project parameters based on selected models, datasets, metrics, and tests
 */
function generateParameters() {
    // Get selected model
    const selectedModel = document.querySelector('input[name="selected_model"]:checked');
    const customModel = document.getElementById('custom_model').value.trim();

    // Get selected datasets and their metrics
    const selectedDatasets = Array.from(document.querySelectorAll('input[name="datasets"]:checked'));
    const datasetMetricAssociations = JSON.parse(document.getElementById('dataset-metric-associations').value || '{}');

    // Get model parameters from step 1
    let modelParameters = {
        model: {
            from_pretrained: {},
            inference: {}
        },
        tokenizer: {
            from_pretrained: {},
            inference: {}
        }
    };

    const modelParametersTextarea = document.getElementById('model_parameters');
    if (modelParametersTextarea && modelParametersTextarea.value.trim()) {
        try {
            const parsedModelParams = JSON.parse(modelParametersTextarea.value);
            if (parsedModelParams.model && parsedModelParams.tokenizer) {
                modelParameters = parsedModelParams;
            }
        } catch (e) {
            console.warn('Error parsing model parameters, using defaults:', e);
        }
    }

    // Get dataset parameters from step 2
    let datasetParameters = {};
    const datasetParametersTextarea = document.getElementById('dataset_parameters');
    if (datasetParametersTextarea && datasetParametersTextarea.value.trim()) {
        try {
            const parsedDatasetParams = JSON.parse(datasetParametersTextarea.value);
            datasetParameters = parsedDatasetParams;
        } catch (e) {
            console.warn('Error parsing dataset parameters, using defaults:', e);
        }
    }

    // Get task parameters from step 4
    let taskType = "";
    let taskParameters = {};
    const taskParametersTextarea = document.getElementById('task_parameters');
    if (taskParametersTextarea && taskParametersTextarea.value.trim()) {
        try {
            const parsedTaskParams = JSON.parse(taskParametersTextarea.value);
            if (parsedTaskParams.task_type) {
                taskType = parsedTaskParams.task_type;
                taskParameters = parsedTaskParams.task_parameters || {};
            }
        } catch (e) {
            console.warn('Error parsing task parameters, using defaults:', e);
        }
    }

    // Build parameters object
    const parameters = {
        task_type: taskType,
        task_parameters: taskParameters,
        model: modelParameters.model,
        tokenizer: modelParameters.tokenizer,
        dataset: {},
        metrics: {}
    };

    // Populate dataset and metrics sections
    selectedDatasets.forEach(datasetCheckbox => {
        const datasetName = datasetCheckbox.value;
        const metrics = datasetMetricAssociations[datasetName] || [];

        // Add dataset configuration - use parameters from step 2 if available
        if (datasetParameters[datasetName]) {
            parameters.dataset[datasetName] = datasetParameters[datasetName];
        } else {
            // Fallback to defaults
            parameters.dataset[datasetName] = {
                input_column: "",
                reference_column: "",
                split: "train",
                num_samples: 15
            };
        }

        // Add metrics configuration
        parameters.metrics[datasetName] = {};
        metrics.forEach(metric => {
            parameters.metrics[datasetName][metric] = {};

            // Add default parameters for specific metrics
            if (metric === 'sari' && parameters.dataset[datasetName]) {
                parameters.metrics[datasetName][metric].sources_column = parameters.dataset[datasetName].input_column || '';
            }
        });
    });

    // Update the textarea
    const parametersTextarea = document.getElementById('parameters');
    if (parametersTextarea) {
        parametersTextarea.value = JSON.stringify(parameters, null, 4);
        validateParametersJSON();
    }
}

/**
 * Resets parameters to auto-generated template based on current selections
 */
function resetParameters() {
    // Regenerate parameters based on current user selections
    generateParameters();
}

/**
 * Validates the JSON in the parameters textarea
 */
function validateParametersJSON() {
    const parametersTextarea = document.getElementById('parameters');
    const jsonValidation = document.getElementById('json-validation');
    const jsonValid = document.getElementById('json-valid');
    const jsonInvalid = document.getElementById('json-invalid');
    const errorMessage = document.getElementById('json-error-message');

    if (!parametersTextarea || !jsonValidation) return;

    const value = parametersTextarea.value.trim();

    if (!value) {
        // Hide validation indicators if empty
        jsonValidation.classList.add('hidden');
        errorMessage.classList.add('hidden');
        return;
    }

    try {
        JSON.parse(value);
        // Valid JSON
        jsonValidation.classList.remove('hidden');
        jsonValid.classList.remove('hidden');
        jsonInvalid.classList.add('hidden');
        errorMessage.classList.add('hidden');
        parametersTextarea.classList.remove('border-red-300');
        parametersTextarea.classList.add('border-gray-300');
    } catch (e) {
        // Invalid JSON
        jsonValidation.classList.remove('hidden');
        jsonValid.classList.add('hidden');
        jsonInvalid.classList.remove('hidden');
        errorMessage.classList.remove('hidden');
        errorMessage.textContent = `JSON Error: ${e.message}`;
        parametersTextarea.classList.remove('border-gray-300');
        parametersTextarea.classList.add('border-red-300');
    }
}

/**
 * Updates the visual selection indicator for metrics (per-dataset)
 */
function updateMetricSelection(checkboxInput, datasetName) {
    const selectionBar = checkboxInput.closest('label').querySelector('.metric-selection-bar');

    if (checkboxInput.checked) {
        selectionBar.classList.remove('bg-gray-300');
        selectionBar.classList.add('bg-primary');
    } else {
        selectionBar.classList.remove('bg-primary');
        selectionBar.classList.add('bg-gray-300');
    }

    // Update the dataset-metric associations
    updateDatasetMetricAssociations();
}

/**
 * Updates the display of selected datasets
 */
function updateSelectedDatasetsDisplay() {
    const selectedDatasets = Array.from(document.querySelectorAll('input[name="datasets"]:checked'));
    const selectedList = document.getElementById('selected-datasets-list');

    if (selectedDatasets.length === 0) {
        selectedList.innerHTML = '<p class="text-gray-500 italic">No datasets selected</p>';
    } else {
        const datasetNames = selectedDatasets.map(dataset => {
            const label = dataset.closest('label').querySelector('h4').textContent;
            return `<span class="inline-block bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded mr-2 mb-1">${label}</span>`;
        });
        selectedList.innerHTML = datasetNames.join('');
    }
}

/**
 * Updates available metrics based on selected datasets - now shows per-dataset selection
 */
function updatePerDatasetMetrics() {
    const selectedDatasets = document.querySelectorAll('input[name="datasets"]:checked');
    const metricsContainer = document.getElementById('metrics-container');

    // Save current metric selections before clearing
    const currentSelections = {};
    const existingMetrics = document.querySelectorAll('input[name="dataset_metrics"]:checked');
    existingMetrics.forEach(metric => {
        const datasetValue = metric.getAttribute('data-dataset');
        if (!currentSelections[datasetValue]) {
            currentSelections[datasetValue] = [];
        }
        currentSelections[datasetValue].push(metric.value);
    });

    // Clear existing content
    metricsContainer.innerHTML = '';

    if (selectedDatasets.length === 0) {
        // Show no metrics message if no datasets selected
        const noMetricsDiv = document.createElement('div');
        noMetricsDiv.id = 'no-metrics-message';
        noMetricsDiv.className = 'px-6 py-8 text-center text-gray-500';
        noMetricsDiv.innerHTML = `
            <i class="fas fa-chart-line text-4xl mb-4 text-gray-300"></i>
            <p>Select a dataset first</p>
            <p class="text-sm">Metrics will appear based on your dataset selection.</p>
        `;
        metricsContainer.appendChild(noMetricsDiv);
    } else {
        // Create sections for each selected dataset
        selectedDatasets.forEach((datasetCheckbox, index) => {
            const datasetValue = datasetCheckbox.value;
            const datasetLabel = datasetCheckbox.closest('label').querySelector('h4').textContent;

            // Create dataset section
            const datasetSection = document.createElement('div');
            datasetSection.className = 'dataset-metrics-section border-b border-gray-100 last:border-b-0';

            // Dataset header
            const datasetIcon = document.getElementById('dataset-color-' + datasetValue);

            // Extract the color class from the original dataset icon
            let iconColorClass = 'text-blue-500'; // fallback
            if (datasetIcon) {
                const colorClass = Array.from(datasetIcon.classList).find(cls => cls.startsWith('text-') && cls.endsWith('-500'));
                if (colorClass) {
                    iconColorClass = colorClass;
                }
            }

            const datasetHeader = document.createElement('div');
            datasetHeader.className = 'px-6 py-3 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200 sticky top-0 z-10';
            datasetHeader.innerHTML = `
                <h4 class="font-medium text-gray-700 flex items-center">
                    <i class="fa-solid fa-database ${iconColorClass} mr-2"></i>
                    ${datasetLabel}
                </h4>
                <p class="text-xs text-gray-500 mt-1">Select metrics for this dataset</p>
            `;
            datasetSection.appendChild(datasetHeader);

            // Get metrics for this dataset
            const metrics = getMetricsForDataset(datasetValue);
            const metricColors = ['text-indigo-500', 'text-emerald-500', 'text-amber-500', 'text-rose-500', 'text-cyan-500'];

            // Create metric items for this dataset
            metrics.forEach((metric, metricIndex) => {
                const colorClass = metricColors[metricIndex % metricColors.length];
                const metricItem = createDatasetMetricItem(metric, colorClass, datasetValue);
                datasetSection.appendChild(metricItem);
            });

            metricsContainer.appendChild(datasetSection);
        });

        // Restore previous selections after DOM is updated
        setTimeout(() => {
            Object.entries(currentSelections).forEach(([datasetValue, metricNames]) => {
                metricNames.forEach(metricName => {
                    const metricCheckbox = document.querySelector(`input[name="dataset_metrics"][data-dataset="${datasetValue}"][value="${metricName}"]`);
                    if (metricCheckbox) {
                        metricCheckbox.checked = true;
                        // Update visual indicator
                        const selectionBar = metricCheckbox.closest('label').querySelector('.metric-selection-bar');
                        if (selectionBar) {
                            selectionBar.classList.remove('bg-gray-300');
                            selectionBar.classList.add('bg-primary');
                        }
                    }
                });
            });

            // Update the hidden field with restored selections
            updateDatasetMetricAssociations();
        }, 0);
    }
}

/**
 * Gets available metrics for a specific dataset
 */
function getMetricsForDataset(datasetValue) {
    const infoIcon = document.getElementById("dataset-info-icon-" + datasetValue);
    const infoText = infoIcon.getAttribute('data-tooltip-metrics');
    const parsedInfo = JSON.parse(infoText);
    let parsed = parsedInfo.metrics;

    if (!Array.isArray(parsed)) {
        const metricsString = parsed.replace(/'/g, '"');
        parsed = JSON.parse(metricsString);
    }

    if (!parsed.includes('all')) {
        return parsed.map(metric => metricsData[metric]);
    }

    return Object.values(metricsData);
}

/**
 * Creates a metric item for a specific dataset and adds it to the metrics container
 */
function createDatasetMetricItem(metric, colorClass, datasetValue) {
    const metricItem = document.createElement('label');
    metricItem.className = 'metric-item group px-6 py-3 hover:bg-gray-50 transition-colors duration-200 flex items-center justify-between cursor-pointer relative';

    const metricId = `${datasetValue}_${metric.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}`;

    // Format output metrics as a comma-separated list
    const outputMetrics = metric.output_metrics
        ? metric.output_metrics.join(', ')
        : 'N/A';

    // Format output format with proper capitalization
    const outputFormat = metric.output_format
        ? metric.output_format.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        : 'N/A';

    // Paper ref
    const paperRef = metric.paper_ref
        ? `<a href="${metric.paper_ref}" target="_blank" rel="noopener noreferrer" class="text-gray-400 hover:text-blue-500 transition-colors duration-200" title="View Paper"><i class="fas fa-external-link-alt text-sm"></i></a>`
        : '';

    metricItem.innerHTML = `
        <div class="absolute left-0 top-0 bottom-0 w-1 bg-gray-300 transition-colors duration-200 metric-selection-bar"></div>
        <input type="checkbox" name="dataset_metrics" value="${metric.name}" data-dataset="${datasetValue}" id="${metricId}" class="hidden metric-checkbox" onchange="updateMetricSelection(this, '${datasetValue}')">
        <div class="flex items-center">
            <i class="fa-solid fa-chart-line ${colorClass} mr-3 ml-2"></i>
            <div>
                <h4 class="font-medium text-gray-800">${metric.name}</h4>
                <p class="text-sm text-gray-500">${metric.description}</p>
            </div>
        </div>
        <div class="flex items-center gap-4">
            ${paperRef}
            <i class="fa-solid fa-circle-info text-gray-400 hover:text-primary cursor-pointer transition-colors duration-200 tooltip-trigger" 
                data-tooltip-title="Metric Information"
                data-tooltip-info='[{"label": "Formula", "value": "${metric.formula}"}, {"label": "Range", "value": "${metric.range}"}, {"label": "Tasks", "value": "${metric.task_types}"}, {"label": "Output Format", "value": "${outputFormat}"}, {"label": "Output Metrics", "value": "${outputMetrics}"}]'></i>
        </div>
    `;

    return metricItem;
}

/**
 * Toggles the task configuration section
 */
function toggleTaskConfiguration() {
    const content = document.getElementById('task-configuration-content');
    const chevron = document.getElementById('task-configuration-chevron');

    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        chevron.classList.remove('fa-chevron-right');
        chevron.classList.add('fa-chevron-down');
    } else {
        content.classList.add('hidden');
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-right');
    }
}

/**
 * Toggles the custom dataset input section
 */
function toggleCustomDataset() {
    const content = document.getElementById('custom-dataset-content');
    const chevron = document.getElementById('custom-dataset-chevron');

    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        chevron.classList.remove('fa-chevron-right');
        chevron.classList.add('fa-chevron-down');
    } else {
        content.classList.add('hidden');
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-right');
    }
}

/**
 * Loads a custom dataset
 */
function loadCustomDataset() {
    const taskType = document.getElementById('task_type');
    if (!taskType || !taskType.value.trim()) {
        alert('Please select a task type first.');
        if (taskType) taskType.focus();
        return;
    }

    const datasetId = document.getElementById('custom_dataset').value.trim();

    if (!datasetId) {
        alert('Please enter a HuggingFace dataset ID');
        return;
    }

    // Basic validation for HuggingFace dataset ID format
    if (!datasetId.includes('/') || datasetId.split('/').length !== 2) {
        alert('Please enter a valid HuggingFace dataset name');
        return;
    }

    // Check if the dataset already exists in the list
    const existingDatasets = Array.from(document.querySelectorAll('input[name="datasets"]'))
        .map(input => input.value);
    if (existingDatasets.includes(datasetId)) {
        alert('This dataset is already loaded.');
        return;
    }

    // Show loading state
    const loadButton = document.getElementById('load-custom-dataset-btn');
    const originalText = loadButton.innerHTML;
    loadButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-xs"></i><span>Loading...</span>';
    loadButton.disabled = true;

    // First validate dataset exists on HuggingFace
    const selectedTaskType = taskType.value.trim();
    const url = `https://huggingface.co/api/datasets/${datasetId}`;
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Dataset not found on HuggingFace (${response.status})`);
            }

            // If dataset exists, check if the dataset tasks match the selected task type
            const tasks_url = `${apiTaskCustomDataset}?dataset_name=${encodeURIComponent(datasetId)}`;
            return fetch(tasks_url);
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`Error fetching tasks: ${data.error}`);
                throw new Error(data.error);
            }
            if (!data.tasks.includes(selectedTaskType) && data.tasks.length > 0 && selectedTaskType != 'custom') {
                alert(`The selected task type "${selectedTaskType}" is not compatible with the dataset.`);
                // Restore button state before returning
                loadButton.innerHTML = originalText;
                loadButton.disabled = false;
                return { success: false };
            }

            // If both checks pass, proceed with adding the dataset
            // Return the tasks data to pass it along the promise chain
            return { success: true, tasks: data.tasks };
        })
        .then(result => {
            if (!result || !result.success) return; // Exit if previous step returned early

            // Fetch additional dataset information and add to datasetsData
            const datasetMetricsUrl = `${apiDatasetMetrics}?dataset_name=${encodeURIComponent(datasetId)}`;
            const configUrl = `${apiDatasetConfigUrl}?dataset_name=${encodeURIComponent(datasetId)}`;

            return Promise.all([
                fetch(datasetMetricsUrl).then(response => response.json()),
                fetch(configUrl).then(response => response.json()),
                Promise.resolve(result.tasks) // Pass tasks data along
            ]);
        })
        .then(([metricsData, configData, tasks]) => {
            if (!metricsData || !configData) return; // Exit if no data received

            // Extract relevant information from the responses
            const metrics = metricsData.error ? [] : (metricsData.metrics || []);
            const config = configData.error ? {} : configData.config ? JSON.parse(configData.config) : {};
            const splits = config.splits ? Object.keys(config.splits) : ['train'];
            const numSamples = config.splits ? Object.values(config.splits).reduce((sum, split) => sum + (split.num_examples || 0), 0) : 0;

            // Add the custom dataset to the datasetsData dictionary
            datasetsData[datasetId] = {
                name: datasetId,
                description: "Custom dataset from HuggingFace",
                metrics: JSON.parse(metrics),
                num_samples: numSamples,
                splits: splits,
                task_types: tasks || [selectedTaskType] // Use fetched tasks or fallback to selected task
            };

            // Add the dataset to the list dynamically
            addCustomDatasetToList(datasetId);

            // Update metrics display to show per-dataset selection
            updatePerDatasetMetrics();

            // Show success state
            loadButton.innerHTML = '<i class="fa-solid fa-check text-xs"></i><span>Added</span>';
            loadButton.classList.remove('bg-gray-500', 'hover:bg-gray-600');
            loadButton.classList.add('bg-green-500');

            // Clear input
            document.getElementById('custom_dataset').value = '';

            // Restore button after 2 seconds
            setTimeout(() => {
                loadButton.innerHTML = originalText;
                loadButton.disabled = false;
                loadButton.classList.remove('bg-green-500');
                loadButton.classList.add('bg-gray-500', 'hover:bg-gray-600');
            }, 2000);
        })
        .catch(error => {
            console.error('Error:', error);
        })
        .finally(() => {
            // Restore button state if there was an error and button is still in loading state
            if (loadButton.disabled && !loadButton.classList.contains('bg-green-500')) {
                loadButton.innerHTML = originalText;
                loadButton.disabled = false;
            }
        });
}

/**
 * Adds a custom dataset to the dataset list
 */
function addCustomDatasetToList(datasetId) {
    // Find the scrollable dataset container using a more reliable selector
    const datasetList = document.querySelector('#step-3 .overflow-y-auto.divide-y');
    const emptyMessage = datasetList.querySelector('.text-center');

    // Remove empty message if it exists
    if (emptyMessage) {
        emptyMessage.remove();
    }

    // Get dataset data from datasetsData (should be populated by loadCustomDataset)
    const datasetData = datasetsData[datasetId];
    if (!datasetData) {
        console.error('Dataset data not found in datasetsData for:', datasetId);
        return;
    }

    // Create new dataset item
    const datasetItem = document.createElement('label');
    datasetItem.className = 'dataset-item group px-6 py-4 hover:bg-gray-50 transition-colors duration-200 flex items-center justify-between cursor-pointer relative';

    // Extract data from datasetsData
    const metrics = datasetData.metrics || [];
    const splits = datasetData.splits || ['train'];
    const tasks = datasetData.task_types || ['all'];
    const numSamples = datasetData.num_samples || 0;
    const splitsDisplay = splits.join(', ');
    const metricsDisplay = Array.isArray(metrics) ? metrics.join(", ") : metrics;
    const tasksDisplay = Array.isArray(tasks) ? tasks.join(", ") : tasks;

    // Random color for dataset icon
    const randomColor = `text-${['purple', 'blue', 'green', 'yellow', 'red'][Math.floor(Math.random() * 5)]}-500`;

    // Create HTML with actual data
    datasetItem.innerHTML = `
        <div class="absolute left-0 top-0 bottom-0 w-1 bg-gray-300 transition-colors duration-200 dataset-selection-bar"></div>
        <input type="checkbox" name="datasets" value="${datasetId}" class="hidden dataset-checkbox" data-splits='${JSON.stringify(splits)}' onchange="updateDatasetSelection(this)">
        <div class="flex items-center">
            <i id="dataset-color-${datasetId}" class="fa-solid fa-database ${randomColor} mr-3 ml-2"></i>
            <div>
                <h4 class="font-medium text-gray-800">${datasetId}</h4>
                <p class="text-sm text-gray-500">${datasetData.description || "Custom dataset from HuggingFace"}</p>
            </div>
        </div>
        <div class="flex items-center gap-4">
            <a href="https://huggingface.co/datasets/${datasetId}" 
                target="_blank" 
                rel="noopener noreferrer"
                class="text-gray-400 hover:text-blue-500 transition-colors duration-200"
                title="View on Hugging Face">
                <i class="fas fa-external-link-alt text-sm"></i>
            </a>
            <i id="dataset-info-icon-${datasetId}"
                class="fa-solid fa-circle-info text-gray-400 hover:text-primary cursor-pointer transition-colors duration-200 tooltip-trigger" 
                data-tooltip-title="Dataset Information"
                data-tooltip-info='[{"label": "Splits", "value": "${splitsDisplay}"}, {"label": "Samples", "value": "${numSamples}"}, {"label": "Tasks", "value": "${tasksDisplay}"}, {"label": "Metrics", "value": "${metricsDisplay}"}]'
                data-tooltip-metrics='{"metrics": ${JSON.stringify(metrics)}}'></i>
        </div>
    `;

    // Insert at the top of the datasets list
    datasetList.insertBefore(datasetItem, datasetList.firstChild);

    // Apply current task filter to the new dataset
    const currentTask = document.getElementById('task_type')?.value;
    if (currentTask) {
        filterDatasetsByTaskType(currentTask);
    }
}

/**
 * Updates the hidden field with dataset-metric associations for form submission
 */
function updateDatasetMetricAssociations() {
    const associations = {};

    // Get all selected dataset checkboxes
    const selectedDatasets = document.querySelectorAll('input[name="datasets"]:checked');

    selectedDatasets.forEach(datasetCheckbox => {
        const datasetValue = datasetCheckbox.value;
        associations[datasetValue] = [];

        // Find all selected metrics for this dataset
        const datasetMetrics = document.querySelectorAll(`input[name="dataset_metrics"][data-dataset="${datasetValue}"]:checked`);
        datasetMetrics.forEach(metricCheckbox => {
            associations[datasetValue].push(metricCheckbox.value);
        });
    });

    document.getElementById('dataset-metric-associations').value = JSON.stringify(associations);
}

/**
 * Toggles the dataset parameters section
 */
function toggleDatasetParameters() {
    const content = document.getElementById('dataset-parameters-content');
    const chevron = document.getElementById('dataset-parameters-chevron');

    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        chevron.classList.remove('fa-chevron-right');
        chevron.classList.add('fa-chevron-down');
    } else {
        content.classList.add('hidden');
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-right');
    }
}

/**
 * Updates dataset parameter forms based on selected datasets
 */
function updateDatasetParametersForms() {
    const selectedDatasets = Array.from(document.querySelectorAll('input[name="datasets"]:checked'));
    const formsContainer = document.getElementById('dataset-parameters-forms');
    const noDatasets = document.getElementById('no-datasets-selected');

    if (selectedDatasets.length === 0) {
        // Show "no datasets selected" message
        noDatasets.classList.remove('hidden');
        // Hide any existing forms
        const existingForms = formsContainer.querySelectorAll('.dataset-parameter-form');
        existingForms.forEach(form => form.remove());
    } else {
        // Hide "no datasets selected" message
        noDatasets.classList.add('hidden');

        // Remove existing forms first
        const existingForms = formsContainer.querySelectorAll('.dataset-parameter-form');
        existingForms.forEach(form => form.remove());

        // Create forms for each selected dataset
        selectedDatasets.forEach((datasetCheckbox, index) => {
            const datasetName = datasetCheckbox.value;
            let datasetSplits = datasetCheckbox.dataset.splits || "['train', 'validation', 'test']";

            // Parse the splits string into an array
            try {
                // Handle both string representation of array and actual array
                if (typeof datasetSplits === 'string') {
                    // Replace single quotes with double quotes for valid JSON
                    const splitsJsonString = datasetSplits.replace(/'/g, '"');
                    datasetSplits = JSON.parse(splitsJsonString);
                }
            } catch (e) {
                console.warn("Failed to parse dataset splits, using defaults:", e);
                datasetSplits = ['train', 'validation', 'test'];
            }

            const formHtml = createDatasetParameterForm(datasetName, index, datasetSplits);
            noDatasets.insertAdjacentHTML('afterend', formHtml);
        });
    }

    // Update JSON preview
    generateDatasetParametersJSON();
}

/**
 * Creates HTML for a dataset parameter form
 */
function createDatasetParameterForm(datasetName, index, datasetSplits = ['train', 'validation', 'test']) {
    const colorClasses = ['text-blue-500', 'text-green-500', 'text-purple-500', 'text-orange-500', 'text-pink-500'];
    const colorClass = colorClasses[index % colorClasses.length];

    // Determine if we should use free-form input (when splits are uncertain/default)
    // This happens when config couldn't be retrieved and defaults to ['train']
    const isUncertainSplits = datasetSplits.length === 1 && datasetSplits[0] === 'train';

    // Generate split field HTML - either dropdown or text input
    let splitFieldHTML;
    if (isUncertainSplits) {
        // Use free-form text input when splits are uncertain
        splitFieldHTML = `
                    <input type="text" id="dataset_split_${datasetName}" name="dataset_split_${datasetName}" 
                            value="" placeholder="e.g., train, test, validation, data"
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            oninput="generateDatasetParametersJSON()">
                    <span class="split-field-indicator text-xs px-2 py-1 rounded bg-orange-100 text-orange-800 inline-block mt-1">
                        <i class="fa-solid fa-pencil mr-1"></i>Free-form entry (no verified splits available)
                    </span>
                    <p class="text-xs text-gray-500 mt-1">Enter the split name manually</p>`;
    } else {
        // Use dropdown when splits are known
        const splitOptions = datasetSplits.map(split =>
            `<option value="${split}">${split}</option>`
        ).join('');
        splitFieldHTML = `
                    <select id="dataset_split_${datasetName}" name="dataset_split_${datasetName}" 
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            onchange="generateDatasetParametersJSON()">
                        ${splitOptions}
                    </select>
                    <span class="split-field-indicator text-xs px-2 py-1 rounded bg-green-100 text-green-800 inline-block mt-1">
                        <i class="fa-solid fa-check-circle mr-1"></i>Verified splits
                    </span>
                    <p class="text-xs text-gray-500 mt-1">Available: ${datasetSplits.join(', ')}</p>`;
    }

    // Check current task type to determine if reference field is needed
    const taskType = document.getElementById('task_type')?.value || '';
    const needsReferenceField = taskType === 'summarization' || taskType === 'text-simplification';
    const needsQueryIdField = taskType === 'text-retrieval';

    // Conditionally build the reference field HTML
    const referenceFieldHTML = needsReferenceField ? `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Reference Column</label>
                    <input type="text" id="dataset_reference_column_${datasetName}" name="dataset_reference_column_${datasetName}" 
                        value="" placeholder="reference"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateDatasetParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Column containing reference/ground truth data</p>
                </div>` : '';

    // Conditionally build the query ID field HTML
    const queryIdFieldHTML = needsQueryIdField ? `
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Query ID Column</label>
                    <input type="text" id="dataset_query_id_column_${datasetName}" name="dataset_query_id_column_${datasetName}" 
                        value="" placeholder="query_id"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateDatasetParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Column containing query identifiers</p>
                </div>` : '';

    return `
        <div class="dataset-parameter-form border border-gray-200 rounded-lg p-4 mb-4" data-dataset="${datasetName}">
            <div class="flex items-center gap-2 mb-4">
                <i class="fa-solid fa-database ${colorClass}"></i>
                <h3 class="text-md font-medium text-gray-700">${datasetName}</h3>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Data Split</label>
                    ${splitFieldHTML}
                </div>
                
                ${queryIdFieldHTML}

                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Input Column</label>
                    <input type="text" id="dataset_input_column_${datasetName}" name="dataset_input_column_${datasetName}" 
                        value="" placeholder="input"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateDatasetParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Column containing input data</p>
                </div>

                ${referenceFieldHTML}
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Num. Samples</label>
                    <input type="number" id="dataset_num_samples_${datasetName}" name="dataset_num_samples_${datasetName}" 
                        value="10" placeholder="10" min="1"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateDatasetParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Max samples to use (leave empty for all)</p>
                </div>
            </div>
        </div>
    `;
}

/**
 * Generates dataset parameters JSON from form inputs
 */
function generateDatasetParametersJSON() {
    const selectedDatasets = Array.from(document.querySelectorAll('input[name="datasets"]:checked'));
    const parameters = {};

    selectedDatasets.forEach(datasetCheckbox => {
        const datasetName = datasetCheckbox.value;

        // Get the split element and check its type
        const splitElement = document.getElementById(`dataset_split_${datasetName}`);
        let split = splitElement?.value || '';
        // Only default to 'train' for dropdowns (select elements), not for text inputs
        if (splitElement && splitElement.tagName === 'SELECT' && !split) {
            split = 'train';
        }

        const inputColumn = document.getElementById(`dataset_input_column_${datasetName}`)?.value || '';
        const referenceColumn = document.getElementById(`dataset_reference_column_${datasetName}`)?.value || '';
        const queryIdColumn = document.getElementById(`dataset_query_id_column_${datasetName}`)?.value || '';
        const numSamplesInput = document.getElementById(`dataset_num_samples_${datasetName}`);
        const numSamples = numSamplesInput?.value ? parseInt(numSamplesInput.value, 10) : 15;

        parameters[datasetName] = {
            split: split,
            input_column: inputColumn,
            num_samples: numSamples
        };

        // Only add reference_column if it exists and has a value
        if (referenceColumn) {
            parameters[datasetName].reference_column = referenceColumn;
        }

        // Only add query_id_column if it exists and has a value
        if (queryIdColumn) {
            parameters[datasetName].query_id_column = queryIdColumn;
        }
    });

    // Update the textarea
    const textarea = document.getElementById('dataset_parameters');
    if (textarea) {
        textarea.value = JSON.stringify(parameters, null, 2);
        validateDatasetParametersJSON();
    }
}

/**
 * Validates the dataset parameters
 */
function validateDatasetParametersJSON() {
    const textarea = document.getElementById('dataset_parameters');
    const statusElement = document.getElementById('dataset-parameters-status');
    const errorElement = document.getElementById('dataset-parameters-error');
    const errorTextElement = document.getElementById('dataset-parameters-error-text');

    if (!textarea) return;

    const value = textarea.value.trim();

    if (!value) {
        // Hide validation indicators if empty
        if (statusElement) statusElement.textContent = 'Empty';
        if (errorElement) errorElement.classList.add('hidden');
        return;
    }

    try {
        const parsed = JSON.parse(value);

        // Get current task type to determine which fields are required
        const taskType = document.getElementById('task_type')?.value || '';
        const needsReferenceField = taskType === 'summarization' || taskType === 'text-simplification';
        const needsQueryIdField = taskType === 'text-retrieval';

        // Validate structure - each dataset should have split and input_column at minimum
        for (const [datasetName, config] of Object.entries(parsed)) {
            const missingParams = [];
            if (!config.split) missingParams.push('split');
            if (!config.input_column) missingParams.push('input_column');

            // Conditionally check task-specific required fields
            if (needsReferenceField && !config.reference_column) {
                missingParams.push('reference_column');
            }

            if (needsQueryIdField && !config.query_id_column) {
                missingParams.push('query_id_column');
            }

            if (missingParams.length > 0) {
                throw new Error(`Dataset "${datasetName}" missing required parameters: ${missingParams.join(', ')}`);
            }

            // Validate num_samples if present
            if (config.num_samples !== undefined && (typeof config.num_samples !== 'number' || config.num_samples < 1)) {
                throw new Error(`Dataset "${datasetName}" has invalid num_samples (must be a positive number or omitted for all samples)`);
            }
        }

        // Valid JSON and structure
        if (statusElement) {
            statusElement.textContent = 'Valid JSON';
            statusElement.className = 'text-xs px-2 py-1 rounded-full bg-green-100 text-green-800';
        }
        if (errorElement) errorElement.classList.add('hidden');

        textarea.classList.remove('border-red-300');
        textarea.classList.add('border-gray-300');

    } catch (e) {
        // Invalid JSON or structure
        if (statusElement) {
            statusElement.textContent = 'Invalid';
            statusElement.className = 'text-xs px-2 py-1 rounded-full bg-red-100 text-red-800';
        }
        if (errorElement) {
            errorElement.classList.remove('hidden');
            if (errorTextElement) errorTextElement.textContent = e.message;
        }

        textarea.classList.remove('border-gray-300');
        textarea.classList.add('border-red-300');
    }
}

/**
 * Updates task parameters based on selected task type
 */
function updateTaskParameters() {
    const taskType = document.getElementById('task_type').value;
    const container = document.getElementById('task-parameters-container');
    const formsContainer = document.getElementById('task-parameters-forms');
    const textarea = document.getElementById('task_parameters');

    if (!taskType) {
        container.classList.add('hidden');
        formsContainer.innerHTML = '';
        // Make textarea readonly and clear it
        if (textarea) {
            textarea.readOnly = true;
            textarea.classList.add('bg-gray-50');
            textarea.classList.remove('bg-white');
        }
        generateTaskParametersJSON();
        // Show all datasets when no task is selected
        filterDatasetsByTaskType(null);
        return;
    }

    container.classList.remove('hidden');

    // Filter datasets based on selected task type
    filterDatasetsByTaskType(taskType);

    // Clear existing forms
    formsContainer.innerHTML = '';

    // Control textarea readonly based on task type
    if (textarea) {
        if (taskType === 'custom') {
            textarea.readOnly = false;
            textarea.classList.remove("cursor-not-allowed")

            const defaultCustomTask = {
                task_type: "custom",
                task_parameters: {
                    "example_parameter": "example_value"
                }
            };
            textarea.value = JSON.stringify(defaultCustomTask, null, 2);
        } else {
            textarea.readOnly = true;
            textarea.classList.add("cursor-not-allowed")
        }
    }

    // Generate task-specific parameter forms
    const taskFormHTML = generateTaskForm(taskType);
    formsContainer.innerHTML = taskFormHTML;

    // Generate JSON from current form state
    generateTaskParametersJSON();
}

/**
 * Filters datasets based on the selected task type
 */
function filterDatasetsByTaskType(taskType) {
    // Get all dataset items
    const datasetItems = document.querySelectorAll('.dataset-item');
    let visibleCount = 0;

    datasetItems.forEach(item => {
        const checkbox = item.querySelector('input[name="datasets"]');
        if (!checkbox) return;

        const datasetName = checkbox.value;

        // Check if datasetsData is available and contains this dataset
        if (typeof datasetsData !== 'undefined' && datasetsData[datasetName]) {
            const dataset = datasetsData[datasetName];
            const taskTypes = dataset.task_types || [];

            // Show dataset if: 1) custom task (shows all), 2) dataset supports 'all' tasks, 3) task type selected AND dataset supports it
            if (taskType === 'custom' || taskTypes.includes('all') || (taskType && taskTypes.includes(taskType))) {
                item.style.display = '';
                visibleCount++;
            } else {
                item.style.display = 'none';
                // Uncheck hidden datasets
                if (checkbox.checked) {
                    checkbox.checked = false;
                    updateDatasetSelection(checkbox);
                }
            }
        } else {
            // If dataset data is not available, show datasets for custom task or when a task is selected
            if (taskType === 'custom' || taskType) {
                item.style.display = '';
                visibleCount++;
            } else {
                item.style.display = 'none';
                // Uncheck hidden datasets
                if (checkbox.checked) {
                    checkbox.checked = false;
                    updateDatasetSelection(checkbox);
                }
            }
        }
    });

    // Update the empty message if no datasets are visible
    updateDatasetsEmptyMessage(visibleCount, taskType);

    // Update metrics display since dataset selection might have changed
    updatePerDatasetMetrics();

    // Update dataset parameters forms since dataset selection might have changed
    updateDatasetParametersForms();
}

/**
 * Updates the empty message in the datasets container
 */
function updateDatasetsEmptyMessage(visibleCount, taskType) {
    const datasetsContainer = document.querySelector('#step-3 .overflow-y-auto.divide-y');
    if (!datasetsContainer) return;

    let emptyMessage = datasetsContainer.querySelector('.empty-datasets-message');

    if (visibleCount === 0) {
        // Show empty message
        if (!emptyMessage) {
            emptyMessage = document.createElement('div');
            emptyMessage.className = 'empty-datasets-message px-6 py-8 text-center text-gray-500';
            datasetsContainer.appendChild(emptyMessage);
        }

        // Update message based on whether a task is selected
        if (!taskType) {
            emptyMessage.innerHTML = `
                <i class="fas fa-database text-4xl mb-4 text-gray-300"></i>
                <p>Select a task type first</p>
                <p class="text-sm">Choose a task type above to see available datasets.</p>
            `;
        } else if (taskType === 'custom') {
            emptyMessage.innerHTML = `
                <i class="fas fa-database text-4xl mb-4 text-gray-300"></i>
                <p>No datasets found</p>
                <p class="text-sm">Load a custom dataset or check your configuration.</p>
            `;
        } else {
            emptyMessage.innerHTML = `
                <i class="fas fa-database text-4xl mb-4 text-gray-300"></i>
                <p>No datasets available for "${taskType}" task</p>
                <p class="text-sm">Try selecting a different task type or load a custom dataset.</p>
            `;
        }
    } else {
        // Remove empty message if it exists
        if (emptyMessage) {
            emptyMessage.remove();
        }
    }
}

/**
 * Generates task-specific parameter forms
 */
function generateTaskForm(taskType) {
    switch (taskType) {
        case 'text-retrieval':
            return generateRAGForm();
        case 'summarization':
            return generateSummarizationForm();
        case 'text-simplification':
            return generateSimplificationForm();
        case 'custom':
            return generateCustomForm();
        default:
            return '<p class="text-gray-500">No specific parameters for this task.</p>';
    }
}
/**
 * Generates Retrieval-Augmented Generation (RAG) parameter form
 */
function generateRAGForm() {
    return `
        <div class="space-y-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Sparse Retrieval (k_sparse)</label>
                    <input type="number" id="rag_k_sparse" name="rag_k_sparse" 
                        value="10" step="1" min="1" max="50"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateTaskParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Number of documents for sparse retrieval (1-50)</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Dense Retrieval (k_dense)</label>
                    <input type="number" id="rag_k_dense" name="rag_k_dense" 
                        value="10" step="1" min="1" max="50"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateTaskParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Number of documents for dense retrieval (1-50)</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Stopword Language</label>
                    <select id="rag_stopword_language" name="rag_stopword_language" 
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            onchange="generateTaskParametersJSON()">
                        <option value="english">English</option>
                    </select>
                    <p class="text-xs text-gray-500 mt-1">Language for stopword removal</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Stemming Language</label>
                    <select id="rag_stemming_language" name="rag_stemming_language" 
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            onchange="generateTaskParametersJSON()">
                        <option value="english">English</option>
                    </select>
                    <p class="text-xs text-gray-500 mt-1">Language for word stemming</p>
                </div>
                
                <div class="md:col-span-2">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Corpus Dataset Name</label>
                    <div class="flex gap-2">
                        <div class="relative flex-1">
                            <input type="text" id="rag_corpus_name" name="rag_corpus_name"
                            value="dtk-llms/mira_rag_corpus"
                            placeholder="rajpurkar/squad"
                            class="w-full pr-10 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent">
                            <a href="https://huggingface.co/datasets/dtk-llms/mira_rag_corpus" id="rag_corpus_link" 
                                class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-blue-500 transition-colors duration-200"
                                target="_blank" rel="noopener noreferrer" title="View on Hugging Face">
                                <i class="fas fa-external-link-alt text-sm"></i>
                            </a>
                        </div>
                        <button type="button" 
                                onclick="validateRAGCorpus()" 
                                class="px-3 py-2 bg-primary text-white rounded-md hover:bg-primary-dark transition-colors duration-200 text-sm">
                            <i class="fa-solid fa-check mr-1"></i>
                            Validate
                        </button>
                    </div>
                    <p class="text-xs text-gray-500 mt-1">HuggingFace dataset name for the RAG corpus.</p>
                    <div id="rag-corpus-validation" class="hidden mt-2"></div>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Corpus Split</label>
                    <select id="rag_corpus_split" name="rag_corpus_split" 
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            onchange="generateTaskParametersJSON()">
                        <option value="train">train</option>
                        <option value="validation">validation</option>
                        <option value="test">test</option>
                    </select>
                    <p class="text-xs text-gray-500 mt-1">Dataset split to use for corpus</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Corpus ID Column</label>
                    <input type="text" id="rag_corpus_id_column" name="rag_corpus_id_column" 
                        value="doc_idx" 
                        placeholder="Column name for document IDs"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateTaskParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Column containing document identifiers</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Corpus Text Column</label>
                    <input type="text" id="rag_corpus_text_column" name="rag_corpus_text_column" 
                        value="doc_text" 
                        placeholder="Column name for document text"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateTaskParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Column containing the document text content</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Metric Cutoff</label>
                    <input type="number" id="rag_metric_cutoff" name="rag_metric_cutoff" 
                        value="0.8" step="0.05" min="0.0" max="1.0"
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        oninput="generateTaskParametersJSON()">
                    <p class="text-xs text-gray-500 mt-1">Relevancy threshold for metrics such as F1, accuracy (0.0 - 1.0)</p>
                </div>
            </div>
            
            <div class="mt-4 pt-4 border-t border-gray-300">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="md:col-span-2">
                        <label class="block text-sm font-medium text-gray-700 mb-1">Qrels Dataset Name</label>
                        <div class="flex gap-2">
                            <div class="relative flex-1">
                                <input type="text" id="rag_qrels_name" name="rag_qrels_name" 
                                    value="dtk-llms/mira_rag_qrels" 
                                    placeholder="beir/trec-covid-qrels"
                                    class="w-full pr-10 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                                    oninput="generateTaskParametersJSON()">
                                    <a href="https://huggingface.co/datasets/dtk-llms/mira_rag_qrels" id="rag_corpus_link" 
                                        class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-blue-500 transition-colors duration-200"
                                        target="_blank" rel="noopener noreferrer" title="View on Hugging Face">
                                        <i class="fas fa-external-link-alt text-sm"></i>
                                    </a>
                            </div>
                            <button type="button" 
                                    onclick="validateRAGQrels()" 
                                    class="px-3 py-2 bg-primary text-white rounded-md hover:bg-primary-dark transition-colors duration-200 text-sm">
                                <i class="fa-solid fa-check mr-1"></i>
                                Validate
                            </button>
                        </div>
                        <p class="text-xs text-gray-500 mt-1">HuggingFace dataset name for relevance judgments.</p>
                        <div id="rag-qrels-validation" class="hidden mt-2"></div>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Qrels Split</label>
                        <select id="rag_qrels_split" name="rag_qrels_split" 
                                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                                onchange="generateTaskParametersJSON()">
                            <option value="train">train</option>
                            <option value="test">test</option>
                            <option value="validation">validation</option>
                        </select>
                        <p class="text-xs text-gray-500 mt-1">Dataset split to use for qrels</p>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Qrels Query ID Column</label>
                        <input type="text" id="rag_qrels_query_id_column" name="rag_qrels_query_id_column" 
                            value="query-id" 
                            placeholder="query-id"
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            oninput="generateTaskParametersJSON()">
                        <p class="text-xs text-gray-500 mt-1">Column containing query IDs</p>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Qrels Corpus ID Column</label>
                        <input type="text" id="rag_qrels_corpus_id_column" name="rag_qrels_corpus_id_column" 
                            value="corpus-id" 
                            placeholder="corpus-id"
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            oninput="generateTaskParametersJSON()">
                        <p class="text-xs text-gray-500 mt-1">Column containing corpus document IDs</p>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Qrels Score Column</label>
                        <input type="text" id="rag_qrels_score_column" name="rag_qrels_score_column" 
                            value="score" 
                            placeholder="score"
                            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                            oninput="generateTaskParametersJSON()">
                        <p class="text-xs text-gray-500 mt-1">Column containing relevance scores</p>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Generates Text Summarization parameter form
 */
function generateSummarizationForm() {
    return `
        <div class="space-y-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">System Message</label>
                <textarea id="sum_system_message" name="sum_system_message" 
                    rows="4"
                    placeholder="Briefly summarise the following paper.\nPaper: <input>\nBrief summary:"
                    class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-vertical"
                    oninput="generateTaskParametersJSON()">Briefly summarise the following paper.\nPaper: <input>\nBrief summary:</textarea>
                <p class="text-xs text-gray-500 mt-1">System message that will be passed to the model before the text to be summarized. Use '&lt;input&gt;' to indicate where the input text will be inserted.</p>
            </div>
        </div>
    `;
}

/**
 * Generates Text Simplification parameter form
 */
function generateSimplificationForm() {
    return `
        <div class="space-y-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">System Message</label>
                <textarea id="simp_system_message" name="simp_system_message" 
                    rows="4"
                    placeholder="Briefly simplify the following paper.\nPaper: <input>\nBrief simplification:"
                    class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-vertical"
                    oninput="generateTaskParametersJSON()">Briefly simplify the following paper.\nPaper: <input>\nBrief simplification:</textarea>
                <p class="text-xs text-gray-500 mt-1">System message that will be passed to the model before the text to be simplified. Use '&lt;input&gt;' to indicate where the input text will be inserted.</p>
            </div>
        </div>
    `;
}

/**
 * Generates Custom Task Parameters form
 */
function generateCustomForm() {
    return `
        <div class="space-y-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
            <p class="text-gray-500">For custom tasks, please define the parameters in the JSON format below.</p>
        </div>
    `;
}

/**
 * Generates task parameters JSON from form inputs
 */
function generateTaskParametersJSON() {
    const taskType = document.getElementById('task_type').value;

    if (!taskType) {
        updateTaskParametersTextarea({ task_type: "", task_parameters: {} });
        return;
    }

    let taskParameters = {};

    switch (taskType) {
        case 'text-retrieval':
            taskParameters = {
                k_sparse: parseInt(document.getElementById('rag_k_sparse')?.value || 10),
                k_dense: parseInt(document.getElementById('rag_k_dense')?.value || 10),
                stopword_language: document.getElementById('rag_stopword_language')?.value || 'english',
                stemming_language: document.getElementById('rag_stemming_language')?.value || 'english',
                corpus_name: document.getElementById('rag_corpus_name')?.value || 'dtk-llms/mira_rag_corpus',
                corpus_split: document.getElementById('rag_corpus_split')?.value || 'train',
                corpus_id_column: document.getElementById('rag_corpus_id_column')?.value || 'doc_id',
                corpus_text_column: document.getElementById('rag_corpus_text_column')?.value || 'doc_text',
                metric_cutoff: parseFloat(document.getElementById('rag_metric_cutoff')?.value || 0.5)
            };

            // Add qrels parameters if qrels_name is provided
            const qrelsName = document.getElementById('rag_qrels_name')?.value;
            if (qrelsName && qrelsName.trim()) {
                taskParameters.qrels_name = qrelsName.trim();
                taskParameters.qrels_split = document.getElementById('rag_qrels_split')?.value || 'test';
                taskParameters.qrels_query_id_column = document.getElementById('rag_qrels_query_id_column')?.value || 'query-id';
                taskParameters.qrels_corpus_id_column = document.getElementById('rag_qrels_corpus_id_column')?.value || 'corpus-id';

                // Score column is optional
                const scoreColumn = document.getElementById('rag_qrels_score_column')?.value;
                if (scoreColumn && scoreColumn.trim()) {
                    taskParameters.qrels_score_column = scoreColumn.trim();
                }

                const minScore = document.getElementById('rag_qrels_min_score')?.value;
                if (minScore && minScore.trim()) {
                    taskParameters.qrels_min_score = parseInt(minScore);
                }
            }
            break;

        case 'summarization':
            taskParameters = {
                system_message: document.getElementById('sum_system_message')?.value.trim() || 'Briefly summarise the following paper.\nPaper: <input>\nBrief summary:'
            };
            break;

        case 'text-simplification':
            taskParameters = {
                system_message: document.getElementById('simp_system_message')?.value.trim() || 'Briefly simplify the following paper.\nPaper: <input>\nBrief simplification:'
            };
            break;

        case 'custom':
            // For custom tasks, don't auto-generate JSON - let users edit manually
            return;
    }

    const taskConfig = {
        task_type: taskType,
        task_parameters: taskParameters
    };

    updateTaskParametersTextarea(taskConfig);
}

/**
 * Updates the task parameters textarea with formatted JSON
 */
function updateTaskParametersTextarea(taskConfig) {
    const textarea = document.getElementById('task_parameters');
    if (textarea) {
        textarea.value = JSON.stringify(taskConfig, null, 2);
        validateTaskParametersJSON();
    }
}

/**
 * Resets the task parameters to their default values
 */
function resetTaskParameters(event) {
    event.preventDefault();

    updateTaskParameters();
}

/**
 * Toggles the task parameters JSON section
 */
function toggleTaskParametersJSON() {
    const content = document.getElementById('task-parameters-json-content');
    const chevron = document.getElementById('task-parameters-json-chevron');

    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        chevron.classList.remove('fa-chevron-right');
        chevron.classList.add('fa-chevron-down');
    } else {
        content.classList.add('hidden');
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-right');
    }
}

/**
 * Validates the task parameters JSON
 */
function validateTaskParametersJSON() {
    const textarea = document.getElementById('task_parameters');
    const statusElement = document.getElementById('task-parameters-status');
    const errorElement = document.getElementById('task-parameters-error');
    const errorTextElement = document.getElementById('task-parameters-error-text');

    if (!textarea) return;

    const value = textarea.value.trim();

    if (!value) {
        if (statusElement) statusElement.textContent = 'Empty';
        if (errorElement) errorElement.classList.add('hidden');
        return;
    }

    try {
        const parsed = JSON.parse(value);

        // Validate structure
        if (!parsed.hasOwnProperty('task_type') || !parsed.hasOwnProperty('task_parameters')) {
            throw new Error('Task configuration must contain "task_type" and "task_parameters" fields');
        }

        // Valid JSON and structure
        if (statusElement) {
            statusElement.textContent = 'Valid JSON';
            statusElement.className = 'text-xs px-2 py-1 rounded-full bg-green-100 text-green-800';
        }
        if (errorElement) errorElement.classList.add('hidden');

        textarea.classList.remove('border-red-300');
        textarea.classList.add('border-gray-300');

    } catch (e) {
        // Invalid JSON or structure
        if (statusElement) {
            statusElement.textContent = 'Invalid JSON';
            statusElement.className = 'text-xs px-2 py-1 rounded-full bg-red-100 text-red-800';
        }
        if (errorElement) {
            errorElement.classList.remove('hidden');
            if (errorTextElement) errorTextElement.textContent = e.message;
        }

        textarea.classList.remove('border-gray-300');
        textarea.classList.add('border-red-300');
    }
}

/**
 * Validates text-retrieval task parameters
 */
function validateRAGTaskParameters() {
    const errors = [];

    // 1. Sparse retrieval k
    const kSparse = document.getElementById('rag_k_sparse');
    if (!kSparse || !kSparse.value.trim() || parseInt(kSparse.value) < 1 || parseInt(kSparse.value) > 50) {
        errors.push('Sparse retrieval k must be between 1 and 50');
    }

    // 2. Dense retrieval k  
    const kDense = document.getElementById('rag_k_dense');
    if (!kDense || !kDense.value.trim() || parseInt(kDense.value) < 1 || parseInt(kDense.value) > 50) {
        errors.push('Dense retrieval k must be between 1 and 50');
    }

    // 3. Stopword language
    const stopwordLang = document.getElementById('rag_stopword_language');
    if (!stopwordLang || !stopwordLang.value.trim()) {
        errors.push('Please select a stopword language');
    }

    // 4. Stemming language
    const stemmingLang = document.getElementById('rag_stemming_language');
    if (!stemmingLang || !stemmingLang.value.trim()) {
        errors.push('Please select a stemming language');
    }

    // 5. Corpus dataset
    const corpusName = document.getElementById('rag_corpus_name');
    if (!corpusName || !corpusName.value.trim()) {
        errors.push('Please provide a corpus dataset name');
    } else if (!corpusName.value.includes('/') || corpusName.value.split('/').length !== 2) {
        errors.push('Corpus dataset must be in format "owner/dataset" (e.g., "rajpurkar/squad")');
    }

    // 6. Corpus split
    const corpusSplit = document.getElementById('rag_corpus_split');
    if (!corpusSplit || !corpusSplit.value.trim()) {
        errors.push('Please select a corpus split');
    }

    // 7. Corpus ID column
    const corpusIdColumn = document.getElementById('rag_corpus_id_column');
    if (!corpusIdColumn || !corpusIdColumn.value.trim()) {
        errors.push('Please provide a corpus ID column name');
    }

    // 8. Corpus text column
    const corpusTextColumn = document.getElementById('rag_corpus_text_column');
    if (!corpusTextColumn || !corpusTextColumn.value.trim()) {
        errors.push('Please provide a corpus text column name');
    }

    // 9. Metric cutoff
    const metricCutoff = document.getElementById('rag_metric_cutoff');
    if (!metricCutoff || !metricCutoff.value.trim() || parseFloat(metricCutoff.value) < 0.0 || parseFloat(metricCutoff.value) > 1.0) {
        errors.push('Metric cutoff must be between 0.0 and 1.0');
    }

    // 10. Qrels dataset name
    const qrelsName = document.getElementById('rag_qrels_name');
    if (qrelsName && qrelsName.value.trim()) {
        if (!qrelsName.value.includes('/') || qrelsName.value.split('/').length !== 2) {
            errors.push('Qrels dataset must be in format "owner/dataset" (e.g., "beir/trec-covid/qrels")');
        }
    }

    // 11. Qrels split
    const qrelsSplit = document.getElementById('rag_qrels_split');
    if (!qrelsSplit || !qrelsSplit.value.trim()) {
        errors.push('Please select a qrels split');
    }

    // 12. Qrels query ID column
    const qrelsQueryIdColumn = document.getElementById('rag_qrels_query_id_column');
    if (!qrelsQueryIdColumn || !qrelsQueryIdColumn.value.trim()) {
        errors.push('Please provide a qrels query ID column name');
    }

    // 13. Qrels corpus ID column
    const qrelsCorpusIdColumn = document.getElementById('rag_qrels_corpus_id_column');
    if (!qrelsCorpusIdColumn || !qrelsCorpusIdColumn.value.trim()) {
        errors.push('Please provide a qrels corpus ID column name');
    }

    // 14. Qrels score column (optional)
    // The score column is optional and will be validated only if provided

    if (errors.length > 0) {
        alert('Text-retrieval task validation errors:\n\n' + errors.join('\n'));
        return false;
    }

    return true;
}

/**
 * Validates summarization task parameters
 */
function validateSummarizationTaskParameters() {
    // Summarization has no additional parameters to validate
    return true;
}

/**
 * Validates text-simplification task parameters
 */
function validateTextSimplificationTaskParameters() {
    const errors = [];

    if (errors.length > 0) {
        alert('Text-simplification task validation errors:\n\n' + errors.join('\n'));
        return false;
    }

    return true;
}

/**
 * Validates the RAG corpus dataset by checking if it exists on HuggingFace
 * and retrieving its configuration information
 */
function validateRAGCorpus() {
    const corpusNameInput = document.getElementById('rag_corpus_name');
    const validationDiv = document.getElementById('rag-corpus-validation');

    if (!corpusNameInput || !validationDiv) return;

    const datasetId = corpusNameInput.value.trim();

    if (!datasetId) {
        showRAGCorpusValidation('error', 'Please enter a corpus dataset name.');
        return;
    }

    // Basic validation for HuggingFace dataset ID format
    if (!datasetId.includes('/') || datasetId.split('/').length !== 2) {
        showRAGCorpusValidation('error', 'Please enter a valid HuggingFace dataset ID (e.g., owner/dataset-name).');
        return;
    }

    // Show loading state
    showRAGCorpusValidation('loading', 'Validating corpus dataset...');

    // Check if the dataset exists on HuggingFace
    const url = `https://huggingface.co/api/datasets/${datasetId}`;
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Dataset not found on HuggingFace (${response.status})`);
            }
            return response.json();
        })
        .then(datasetInfo => {
            // Dataset exists, now get configuration
            if (typeof apiDatasetConfigUrl !== 'undefined') {
                const configUrl = `${apiDatasetConfigUrl}?dataset_name=${encodeURIComponent(datasetId)}`;
                return fetch(configUrl)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        let config = {};
                        if (data.config) {
                            try {
                                config = JSON.parse(data.config);
                            } catch (e) {
                                console.warn('Error parsing config JSON:', e);
                                config = data.config; // Use as-is if it's already an object
                            }
                        }

                        if (!config.splits) {
                            // No splits found - convert to free-form text input
                            convertCorpusSplitToTextInput();
                            showRAGCorpusValidation('success', 'Dataset validated successfully! No split information available - please enter the split name manually (e.g., train, test, validation).');
                            generateTaskParametersJSON();
                            return;
                        }

                        const corpusSplits = Object.keys(config.splits);
                        // Update corpus split options if splits are available
                        updateCorpusSplitOptions(corpusSplits);

                        showRAGCorpusValidation('success', `Dataset validated successfully! Found ${corpusSplits.length} splits.`);

                        // Regenerate parameters with validated corpus
                        generateTaskParametersJSON();
                    });
            } else {
                // No API available - convert to free-form text input since we can't verify splits
                convertCorpusSplitToTextInput();
                showRAGCorpusValidation('success', 'Dataset found on HuggingFace. Configuration unavailable - please enter the split name manually.');
                generateTaskParametersJSON();
            }
        })
        .catch(error => {
            console.error('Error validating RAG corpus:', error);
            showRAGCorpusValidation('error', error.message || 'Failed to validate dataset.');
        });
}

/**
 * Validates the RAG qrels dataset by checking if it exists on HuggingFace
 * and retrieving its configuration information
 */
function validateRAGQrels() {
    const qrelsNameInput = document.getElementById('rag_qrels_name');
    const validationDiv = document.getElementById('rag-qrels-validation');

    if (!qrelsNameInput || !validationDiv) return;

    const datasetId = qrelsNameInput.value.trim();

    if (!datasetId) {
        showRAGQrelsValidation('error', 'Please enter a qrels dataset name.');
        return;
    }

    // Basic validation for HuggingFace dataset ID format
    if (!datasetId.includes('/') || datasetId.split('/').length !== 2) {
        showRAGQrelsValidation('error', 'Invalid dataset ID format. Expected format: owner/dataset-name');
        return;
    }

    // Show loading state
    showRAGQrelsValidation('loading', 'Validating qrels dataset...');

    // Check if the dataset exists on HuggingFace
    const url = `https://huggingface.co/api/datasets/${datasetId}`;
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Dataset not found: ${datasetId}`);
            }
            return response.json();
        })
        .then(datasetInfo => {
            // Dataset exists, now get configuration
            if (typeof apiDatasetConfigUrl !== 'undefined') {
                const configUrl = `${apiDatasetConfigUrl}?dataset_name=${encodeURIComponent(datasetId)}`;
                return fetch(configUrl)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Failed to fetch dataset configuration');
                        }
                        return response.json();
                    })
                    .then(data => {
                        let config = {};
                        if (data.config) {
                            try {
                                config = JSON.parse(data.config);
                            } catch (e) {
                                console.warn('Error parsing config JSON:', e);
                                config = data.config; // Use as-is if it's already an object
                            }
                        } else if (data.error) {
                            console.warn('Dataset config error:', data.error);
                        }

                        if (!config.splits) {
                            // No splits found in config - convert to free-form text input
                            convertQrelsSplitToTextInput();
                            showRAGQrelsValidation('success', 'Dataset found on HuggingFace. Configuration unavailable - please enter the split name manually.');
                            generateTaskParametersJSON();
                            return;
                        }

                        const qrelsSplits = Object.keys(config.splits);
                        // Update qrels split options if splits are available
                        updateQrelsSplitOptions(qrelsSplits);

                        showRAGQrelsValidation('success', `Dataset validated successfully! Found ${qrelsSplits.length} splits.`);

                        // Regenerate parameters with validated qrels
                        generateTaskParametersJSON();
                    });
            } else {
                // No API available - convert to free-form text input since we can't verify splits
                convertQrelsSplitToTextInput();
                showRAGQrelsValidation('success', 'Dataset found on HuggingFace. Configuration unavailable - please enter the split name manually.');
                generateTaskParametersJSON();
            }
        })
        .catch(error => {
            showRAGQrelsValidation('error', `Dataset validation failed: ${error.message}`);
        });
}

/**
 * Updates the qrels split dropdown options based on available splits
 */
function updateQrelsSplitOptions(splits) {
    const container = document.getElementById('rag_qrels_split')?.parentElement;
    if (!container) return;

    // Remove any existing input/select and indicator
    const oldInput = container.querySelector('#rag_qrels_split');
    const oldIndicator = container.querySelector('.split-field-indicator');
    if (oldInput) oldInput.remove();
    if (oldIndicator) oldIndicator.remove();

    const currentValue = oldInput?.value || '';

    // Create new select dropdown
    const selectElement = document.createElement('select');
    selectElement.id = 'rag_qrels_split';
    selectElement.name = 'rag_qrels_split';
    selectElement.className = 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent';
    selectElement.onchange = generateTaskParametersJSON;

    // Add options
    splits.forEach(split => {
        const option = document.createElement('option');
        option.value = split;
        option.textContent = split;
        selectElement.appendChild(option);
    });

    // Try to restore previous selection, or default to 'train' if available
    if (splits.includes(currentValue)) {
        selectElement.value = currentValue;
    } else if (splits.includes('train')) {
        selectElement.value = 'train';
    } else if (splits.length > 0) {
        selectElement.value = splits[0];
    }

    // Add indicator badge
    const indicator = document.createElement('span');
    indicator.className = 'split-field-indicator text-xs px-2 py-1 rounded bg-green-100 text-green-800 inline-block mt-1';
    indicator.innerHTML = '<i class="fa-solid fa-check-circle mr-1"></i>Verified splits';

    // Insert select and indicator
    const label = container.querySelector('label');
    const helpText = container.querySelector('p.text-xs');
    if (label && helpText) {
        container.insertBefore(selectElement, helpText);
        container.insertBefore(indicator, helpText);
    } else {
        container.appendChild(selectElement);
        container.appendChild(indicator);
    }
}

/**
 * Converts the qrels split select to a text input for free-form entry
 */
function convertQrelsSplitToTextInput() {
    const container = document.getElementById('rag_qrels_split')?.parentElement;
    if (!container) return;

    // Remove any existing input/select and indicator
    const oldInput = container.querySelector('#rag_qrels_split');
    const oldIndicator = container.querySelector('.split-field-indicator');
    if (oldInput) oldInput.remove();
    if (oldIndicator) oldIndicator.remove();

    const currentValue = oldInput?.value || '';

    // Create new text input
    const inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.id = 'rag_qrels_split';
    inputElement.name = 'rag_qrels_split';
    inputElement.value = ''; // Leave blank for user to fill
    inputElement.placeholder = 'e.g., train, test, validation';
    inputElement.className = 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent';
    inputElement.oninput = generateTaskParametersJSON;

    // Add indicator badge
    const indicator = document.createElement('span');
    indicator.className = 'split-field-indicator text-xs px-2 py-1 rounded bg-orange-100 text-orange-800 inline-block mt-1';
    indicator.innerHTML = '<i class="fa-solid fa-pencil mr-1"></i>Free-form entry (no verified splits available)';

    // Insert input and indicator
    const label = container.querySelector('label');
    const helpText = container.querySelector('p.text-xs');
    if (label && helpText) {
        container.insertBefore(inputElement, helpText);
        container.insertBefore(indicator, helpText);
    } else {
        container.appendChild(inputElement);
        container.appendChild(indicator);
    }
}

/**
 * Shows validation feedback for RAG qrels dataset
 */
function showRAGQrelsValidation(type, message) {
    const validationDiv = document.getElementById('rag-qrels-validation');
    if (!validationDiv) return;

    validationDiv.classList.remove('hidden');

    let iconClass, bgClass, textClass;

    switch (type) {
        case 'success':
            iconClass = 'fa-solid fa-check-circle text-green-500';
            bgClass = 'bg-green-50 border-green-200';
            textClass = 'text-green-800';
            break;
        case 'error':
            iconClass = 'fa-solid fa-exclamation-triangle text-red-500';
            bgClass = 'bg-red-50 border-red-200';
            textClass = 'text-red-800';
            break;
        case 'loading':
            iconClass = 'fa-solid fa-spinner fa-spin text-blue-500';
            bgClass = 'bg-blue-50 border-blue-200';
            textClass = 'text-blue-800';
            break;
        default:
            iconClass = 'fa-solid fa-info-circle text-gray-500';
            bgClass = 'bg-gray-50 border-gray-200';
            textClass = 'text-gray-800';
    }

    validationDiv.innerHTML = `
        <div class="p-3 border rounded-md ${bgClass}">
            <div class="flex items-start gap-2">
                <i class="${iconClass} mt-0.5"></i>
                <p class="text-sm ${textClass}">${message}</p>
            </div>
        </div>
    `;
}

/**
 * Updates the corpus split dropdown options based on available splits
 */
function updateCorpusSplitOptions(splits) {
    const container = document.getElementById('rag_corpus_split')?.parentElement;
    if (!container) return;

    // Remove any existing input/select and indicator
    const oldInput = container.querySelector('#rag_corpus_split');
    const oldIndicator = container.querySelector('.split-field-indicator');
    if (oldInput) oldInput.remove();
    if (oldIndicator) oldIndicator.remove();

    const currentValue = oldInput?.value || '';

    // Create new select dropdown
    const selectElement = document.createElement('select');
    selectElement.id = 'rag_corpus_split';
    selectElement.name = 'rag_corpus_split';
    selectElement.className = 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent';
    selectElement.onchange = generateTaskParametersJSON;

    // Add options
    splits.forEach(split => {
        const option = document.createElement('option');
        option.value = split;
        option.textContent = split;
        selectElement.appendChild(option);
    });

    // Try to restore previous selection, or default to 'train' if available
    if (splits.includes(currentValue)) {
        selectElement.value = currentValue;
    } else if (splits.includes('train')) {
        selectElement.value = 'train';
    } else if (splits.length > 0) {
        selectElement.value = splits[0];
    }

    // Add indicator badge
    const indicator = document.createElement('span');
    indicator.className = 'split-field-indicator text-xs px-2 py-1 rounded bg-green-100 text-green-800 inline-block mt-1';
    indicator.innerHTML = '<i class="fa-solid fa-check-circle mr-1"></i>Verified splits';

    // Insert select and indicator
    const label = container.querySelector('label');
    const helpText = container.querySelector('p.text-xs');
    if (label && helpText) {
        container.insertBefore(selectElement, helpText);
        container.insertBefore(indicator, helpText);
    } else {
        container.appendChild(selectElement);
        container.appendChild(indicator);
    }
}

/**
 * Converts the corpus split select to a text input for free-form entry
 */
function convertCorpusSplitToTextInput() {
    const container = document.getElementById('rag_corpus_split')?.parentElement;
    if (!container) return;

    // Remove any existing input/select and indicator
    const oldInput = container.querySelector('#rag_corpus_split');
    const oldIndicator = container.querySelector('.split-field-indicator');
    if (oldInput) oldInput.remove();
    if (oldIndicator) oldIndicator.remove();

    const currentValue = oldInput?.value || '';

    // Create new text input
    const inputElement = document.createElement('input');
    inputElement.type = 'text';
    inputElement.id = 'rag_corpus_split';
    inputElement.name = 'rag_corpus_split';
    inputElement.value = ''; // Leave blank for user to fill
    inputElement.placeholder = 'e.g., train, test, validation';
    inputElement.className = 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent';
    inputElement.oninput = generateTaskParametersJSON;

    // Add indicator badge
    const indicator = document.createElement('span');
    indicator.className = 'split-field-indicator text-xs px-2 py-1 rounded bg-orange-100 text-orange-800 inline-block mt-1';
    indicator.innerHTML = '<i class="fa-solid fa-pencil mr-1"></i>Free-form entry (no verified splits available)';

    // Insert input and indicator
    const label = container.querySelector('label');
    const helpText = container.querySelector('p.text-xs');
    if (label && helpText) {
        container.insertBefore(inputElement, helpText);
        container.insertBefore(indicator, helpText);
    } else {
        container.appendChild(inputElement);
        container.appendChild(indicator);
    }
}

/**
 * Shows validation feedback for RAG corpus dataset
 */
function showRAGCorpusValidation(type, message) {
    const validationDiv = document.getElementById('rag-corpus-validation');
    if (!validationDiv) return;

    validationDiv.classList.remove('hidden');

    let iconClass, bgClass, textClass;

    switch (type) {
        case 'success':
            iconClass = 'fa-solid fa-check-circle text-green-500';
            bgClass = 'bg-green-50 border-green-200';
            textClass = 'text-green-800';
            break;
        case 'error':
            iconClass = 'fa-solid fa-exclamation-triangle text-red-500';
            bgClass = 'bg-red-50 border-red-200';
            textClass = 'text-red-800';
            break;
        case 'loading':
            iconClass = 'fa-solid fa-spinner fa-spin text-blue-500';
            bgClass = 'bg-blue-50 border-blue-200';
            textClass = 'text-blue-800';
            break;
        default:
            iconClass = 'fa-solid fa-info-circle text-gray-500';
            bgClass = 'bg-gray-50 border-gray-200';
            textClass = 'text-gray-800';
    }

    validationDiv.innerHTML = `
        <div class="p-3 border rounded-md ${bgClass}">
            <div class="flex items-start gap-2">
                <i class="${iconClass} mt-0.5"></i>
                <p class="text-sm ${textClass}">${message}</p>
            </div>
        </div>
    `;
}

/**
 * Initialize the wizard when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function () {
    // Initialize wizard
    showStep(currentStep);

    // Initialize model parameters with defaults
    generateDefaultModelParameters();

    // Add parameter textarea event listeners
    const parametersTextarea = document.getElementById('parameters');
    if (parametersTextarea) {
        // Validate JSON on input
        parametersTextarea.addEventListener('input', validateParametersJSON);

        // Auto-generate parameters when step 5 is shown for the first time
        let parametersGenerated = false;
        const originalShowStep = showStep;
        showStep = function (step) {
            originalShowStep(step);
            if (step === 5 && !parametersGenerated) {
                generateParameters();
                parametersGenerated = true;
            }
        };
    }

    // Add model parameters textarea event listener
    const modelParametersTextarea = document.getElementById('model_parameters');
    if (modelParametersTextarea) {
        modelParametersTextarea.addEventListener('input', validateModelParametersJSON);
    }

    // Add dataset parameters textarea event listener
    const datasetParametersTextarea = document.getElementById('dataset_parameters');
    if (datasetParametersTextarea) {
        datasetParametersTextarea.addEventListener('input', validateDatasetParametersJSON);
    }

    // Initialize dataset parameters forms
    updateDatasetParametersForms();

    // Hide all datasets by default when no task is selected
    const initialTaskType = document.getElementById('task_type')?.value;
    filterDatasetsByTaskType(initialTaskType || null);

    // Create a single generic tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'generic-info-tooltip';

    const tooltipHeader = document.createElement('div');
    tooltipHeader.className = 'tooltip-header';

    const tooltipBody = document.createElement('div');
    tooltipBody.className = 'tooltip-body';

    tooltip.appendChild(tooltipHeader);
    tooltip.appendChild(tooltipBody);
    document.body.appendChild(tooltip);

    // Handle tooltip triggers
    document.addEventListener('mouseenter', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('tooltip-trigger')) {
            const title = e.target.getAttribute('data-tooltip-title') || 'Information';
            const infoData = e.target.getAttribute('data-tooltip-info');

            // Set header
            tooltipHeader.textContent = title;

            // Create structured content from data
            let content = '';
            try {
                if (infoData) {
                    const info = JSON.parse(infoData);
                    info.forEach(item => {
                        // Parse array-like strings for better display
                        let displayValue = item.value;

                        // Check if the value looks like a stringified array
                        if (typeof displayValue === 'string' &&
                            ((displayValue.startsWith('[') && displayValue.endsWith(']')) || (displayValue.startsWith("['") && displayValue.endsWith("']")))) {
                            try {
                                // Try to parse as JSON first
                                const parsed = JSON.parse(displayValue);
                                if (Array.isArray(parsed)) {
                                    displayValue = parsed.join(', ');
                                }
                            } catch {
                                // If JSON parsing fails, try to extract values manually
                                // Handle formats like "['test', 'train']" or "[test, train]"
                                const match = displayValue.match(/\[(.*)\]/);
                                if (match) {
                                    const innerContent = match[1];
                                    // Split by comma and clean up quotes
                                    const items = innerContent.split(',').map(item =>
                                        item.trim().replace(/^['"]|['"]$/g, '')
                                    );
                                    displayValue = items.join(', ');
                                }
                            }
                        }

                        content += `
                            <div class="info-item">
                                <span class="info-label">${item.label}:</span> 
                                <span class="info-value">${displayValue}</span>
                            </div>
                        `;
                    });
                }
            } catch (error) {
                // Fallback for old format (model tooltips)
                const size = e.target.getAttribute('data-model-size');
                const architecture = e.target.getAttribute('data-model-architecture');
                const vocab_size = e.target.getAttribute('data-model-vocab');

                if (size || architecture || vocab_size) {
                    tooltipHeader.textContent = 'Model Information';
                    content = `
                        <div class="info-item">
                            <span class="info-label">Size:</span> 
                            <span class="info-value">${size || 'Unknown'} MB</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Architecture:</span> 
                            <span class="info-value">${architecture || 'Unknown'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Vocab Size:</span> 
                            <span class="info-value">${vocab_size || 'Unknown'}</span>
                        </div>
                    `;
                }
            }

            tooltipBody.innerHTML = content || '<div class="info-item">No information available</div>';
            tooltip.style.display = 'block';

            // Position tooltip to the right of the trigger
            const rect = e.target.getBoundingClientRect();
            tooltip.style.left = (rect.right + 32) + 'px';
            tooltip.style.top = (rect.top + rect.height / 2 - tooltip.offsetHeight / 2) + 'px';
        }
    }, true);

    document.addEventListener('mouseleave', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('tooltip-trigger')) {
            tooltip.style.display = 'none';
        }
    }, true);

    // Create a single test metadata tooltip element
    const testTooltip = document.createElement('div');
    testTooltip.className = 'test-metadata-tooltip';

    const testTooltipHeader = document.createElement('div');
    testTooltipHeader.className = 'tooltip-header';

    const testTooltipBody = document.createElement('div');
    testTooltipBody.className = 'tooltip-body';

    const testTooltipFooter = document.createElement('div');
    testTooltipFooter.className = 'tooltip-footer';

    testTooltip.appendChild(testTooltipHeader);
    testTooltip.appendChild(testTooltipBody);
    testTooltip.appendChild(testTooltipFooter);
    document.body.appendChild(testTooltip);

    // Handle test metadata tooltip triggers
    document.addEventListener('mouseenter', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('test-metadata-trigger')) {
            const metadataStr = e.target.getAttribute('data-test-metadata');

            let metadata = {};

            try {
                // Try to parse as JSON first
                if (metadataStr && metadataStr.trim() !== '' && metadataStr !== '{}') {
                    // Handle case where metadata might already be an object (dict from Django)
                    if (typeof metadataStr === 'string') {
                        // Clean up common escaping issues
                        let cleanedStr = metadataStr.trim();

                        // If it doesn't start with {, it might be malformed
                        if (!cleanedStr.startsWith('{')) {
                            console.warn('Metadata does not appear to be valid JSON:', cleanedStr);
                            metadata = {};
                        } else {
                            metadata = JSON.parse(cleanedStr);
                        }
                    } else {
                        metadata = metadataStr;
                    }
                }
            } catch (error) {
                metadata = {};
            }

            // Header with title and tags
            let headerContent = '<span class="header-title">Method Information</span>';
            if (metadata.tags && Array.isArray(metadata.tags) && metadata.tags.length > 0) {
                headerContent += '<div class="tags-container">';
                metadata.tags.forEach(tag => {
                    headerContent += `<span class="tag">${tag}</span>`;
                });
                headerContent += '</div>';
            }
            testTooltipHeader.innerHTML = headerContent;

            // Body with paper title and description
            let bodyContent = '';

            // Paper title first
            if (metadata.paper_name) {
                bodyContent += `<div class="paper-title">${metadata.paper_name}</div>`;
            }

            // Description
            if (metadata.description && metadata.description.trim() !== '') {
                bodyContent += `<div class="description">${metadata.description}</div>`;
            } else {
                bodyContent += '<div class="description">No description available</div>';
            }

            testTooltipBody.innerHTML = bodyContent;

            // Footer with reference, publisher, and booktitle
            let footerContent = '';
            if (metadata.paper_ref) {
                footerContent = `<div class="reference-info">${metadata.paper_ref}</div>`;
            } else {
                footerContent = '<div class="reference-info">No reference information available</div>';
            }

            // Add publisher and booktitle as subtle secondary information
            const secondaryInfo = [];
            if (metadata.publisher) {
                secondaryInfo.push(metadata.publisher);
            }
            if (metadata.booktitle) {
                secondaryInfo.push(metadata.booktitle);
            }

            if (secondaryInfo.length > 0) {
                footerContent += `<div class="text-xs text-gray-500 mt-1 italic">${secondaryInfo.join(' • ')}</div>`;
            }

            testTooltipFooter.innerHTML = footerContent;

            testTooltip.style.display = 'block';

            // Position tooltip to the right of the trigger with more spacing
            const rect = e.target.getBoundingClientRect();
            testTooltip.style.left = (rect.right + 32) + 'px';
            testTooltip.style.top = (rect.top + rect.height / 2 - testTooltip.offsetHeight / 2) + 'px';
        }
    }, true);

    document.addEventListener('mouseleave', function (e) {
        if (e.target && e.target.classList && e.target.classList.contains('test-metadata-trigger')) {
            testTooltip.style.display = 'none';
        }
    }, true);
});
