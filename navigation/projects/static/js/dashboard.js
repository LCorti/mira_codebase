// Function to activate sidebar sections
function activateSidebarSection(section) {
    const sidebarButton = document.querySelector(`button[data-section="${section}"]`);
    if (sidebarButton) {
        sidebarButton.click();
    }
}

class ContentManager {
    constructor(testsData) {
        this.testsData = testsData; // Store the tests data passed from Django
        this.currentContent = null;
        this.contentArea = document.getElementById('content-area');
        this.welcomeScreen = document.getElementById('welcome-screen');
        this.dashboardContent = document.getElementById('dashboard-content');
        
        // Sample navigation properties
        this.currentJob = null;
        this.currentSampleIndex = 0;
        this.totalSamples = 0;
        
        // LXT output token navigation properties
        this.currentOutputTokenIndex = 0;
        this.currentLXTData = null; // Full array of samples
        this.currentLXTSampleIndex = 0; // Which input sample we're viewing
        this.totalLXTSamples = 0;
        this.lockedLXTOutputToken = null; // Index of locked output token (null if none locked)
        
        // SHAP visualization properties
        this.currentSHAPData = null; // Full array of samples
        this.currentSHAPSampleIndex = 0; // Which input sample we're viewing
        this.totalSHAPSamples = 0;
        this.lockedOutputToken = null; // Index of locked output token (null if none locked)
        
        // SHAP Encoder visualization properties
        this.currentSHAPEncoderData = null; // Full array of query-document pairs
        this.currentSHAPEncoderSampleIndex = 0; // Which query we're viewing
        this.currentSHAPEncoderDocIndex = 0; // Which document for current query
        this.totalSHAPEncoderSamples = 0;
        this.totalSHAPEncoderDocs = 0; // Number of documents for current query
        
        // IG Encoder visualization properties
        this.currentIGEncoderData = null; // Full array of query-document pairs
        this.currentIGEncoderSampleIndex = 0; // Which query we're viewing
        this.currentIGEncoderDocIndex = 0; // Which document for current query
        this.totalIGEncoderSamples = 0;
        this.totalIGEncoderDocs = 0; // Number of documents for current query
        
        // Job group state tracking
        this.jobGroupStates = {}; // Store expanded/collapsed state of job groups by parameter hash
    }

    // Helper function to clear all highlights
    clearAllHighlights() {
        // Clear test highlights
        document.querySelectorAll('[id^="test-item-"]').forEach(element => {
            element.classList.remove('bg-gray-200', 'bg-purple-50', 'border-l-4', 'border-purple-500', 'pl-0', 'hover:bg-purple-100');
            if (!element.classList.contains('hover:bg-gray-50')) {
                element.classList.add('hover:bg-gray-50');
            }
        });
        
        // Clear method highlights
        document.querySelectorAll('[id^="method-item-"]').forEach(element => {
            element.classList.remove('bg-gray-200', 'bg-purple-50', 'border-l-4', 'border-purple-500', 'pl-2', 'hover:bg-purple-100');
            if (!element.classList.contains('hover:bg-gray-50')) {
                element.classList.add('hover:bg-gray-50');
            }
        });
    }

    // Helper function to reapply highlight after view switches
    reapplyHighlight() {
        if (window.currentHighlight) {
            const { type, id } = window.currentHighlight;
            const element = document.getElementById(`${type}-item-${id}`);
            if (element) {
                element.classList.add('bg-gray-200');
                element.classList.remove('hover:bg-gray-50');
                element.classList.add('hover:bg-gray-300');
            }
        }
    }

    // Helper function to highlight a specific test
    highlightTest(testId) {
        this.clearAllHighlights();
        const testElement = document.getElementById(`test-item-${testId}`);
        if (testElement) {
            testElement.classList.add('bg-gray-200');
            testElement.classList.remove('hover:bg-gray-50');
            testElement.classList.add('hover:bg-gray-300');
        }
        // Store current highlight state
        window.currentHighlight = { type: 'test', id: testId };
    }

    // Helper function to highlight a specific method
    highlightMethod(methodId) {
        this.clearAllHighlights();
        const methodElement = document.getElementById(`method-item-${methodId}`);
        if (methodElement) {
            methodElement.classList.add('bg-gray-200');
            methodElement.classList.remove('hover:bg-gray-50');
            methodElement.classList.add('hover:bg-gray-300');
        }
        // Store current highlight state
        window.currentHighlight = { type: 'method', id: methodId };
    }

    // Clear job tag highlights
    clearJobTagHighlights() {
        const jobCards = document.querySelectorAll('.job-card-with-tag-highlight');
        jobCards.forEach(card => {
            card.classList.remove('job-card-with-tag-highlight', 'border-blue-500', 'border-2');
        });
        
        // Clear the count indicator
        const countIndicator = document.getElementById('tag-match-count');
        if (countIndicator) {
            countIndicator.remove();
        }
    }

    showTestContent(testId) {
        const testData = this.getTestById(testId);
        if (!testData) return;

        const template = document.getElementById('test-overview-template').innerHTML;
        
        // Calculate summary statistics
        const methodsCount = testData.methods.length;
        const completedRuns = testData.methods.filter(m => m.jobs && m.jobs.length > 0).length;
        const successfulRuns = testData.methods.filter(m => 
            m.jobs && m.jobs.some(job => job.status === 'COMPLETED' || job.status === 'SUCCESS')
        ).length;
        const failedRuns = testData.methods.filter(m => 
            m.jobs && m.jobs.some(job => job.status === 'FAILED')
        ).length;
        const uniqueDatasets = [...new Set(testData.methods.map(m => m.dataset_name))].length;
        const methodIds = testData.methods.map(m => m.id);

        const rendered = this.renderTemplate(template, {
            projectId: testData.project_id,
            testName: testData.name,
            testDescription: testData.description || 'No description available',
            testId: testData.id,
            methodIds: methodIds,
            methodsCount: methodsCount,
            completedRuns: completedRuns,
            successfulRuns: successfulRuns,
            failedRuns: failedRuns,
            uniqueDatasets: uniqueDatasets,
            methods: testData.methods.map(method => {
                const latestJob = method.jobs && method.jobs.length > 0 ? method.jobs[method.jobs.length - 1] : null;
                const hasSuccessfulJob = method.jobs && method.jobs.some(job => job.status === 'COMPLETED' || job.status === 'SUCCESS');
                const hasWarningJob = method.jobs && method.jobs.some(job => job.status === 'WARNING');
                const hasFailedJob = method.jobs && method.jobs.some(job => job.status === 'FAILED');
                
                const hasCanceledJob = method.jobs && method.jobs.some(job => job.status === 'CANCELED' || job.status === 'CANCELLED');
                
                return {
                    id: method.id,
                    name: method.name,
                    description: method.description || 'No description available',
                    dataset: method.dataset_name,
                    metrics: method.metrics.join(', '),
                    jobsCount: method.jobs ? method.jobs.length : 0,
                    hasJobs: method.jobs && method.jobs.length > 0,
                    statusClass: hasSuccessfulJob ? 'text-green-600' : hasWarningJob ? 'text-orange-600' : hasFailedJob ? 'text-red-600' : hasCanceledJob ? 'text-gray-500' : 'text-gray-500',
                    statusText: hasSuccessfulJob ? 'Completed' : hasWarningJob ? 'Warning' : hasFailedJob ? 'Failed' : hasCanceledJob ? 'Canceled' : 'Not Run',
                    statusIcon: hasSuccessfulJob ? 'fa-check-circle' : hasWarningJob ? 'fa-exclamation-triangle' : hasFailedJob ? 'fa-times-circle' : hasCanceledJob ? 'fa-ban' : 'fa-clock',
                    lastJobStatus: latestJob ? latestJob.status : 'No jobs',
                    testId: testData.id
                };
            })
        });
        
        this.switchToContent(rendered);
        this.currentContent = { type: 'test', id: testId };
        
        // Highlight the selected test
        this.highlightTest(testId);
    }

    async showMethodContent(methodId, testId) {
        const testData = this.getTestById(testId);
        const methodData = testData?.methods.find(m => m.id === methodId);

        if (!testData || !methodData) return;

        const template = document.getElementById('method-detail-template').innerHTML;
        
        // Analyze jobs and results
        const latestJob = methodData.jobs && methodData.jobs.length > 0 ? methodData.jobs[methodData.jobs.length - 1] : null;
        const hasSuccessfulJob = methodData.jobs && methodData.jobs.some(job => job.status === 'COMPLETED' || job.status === 'SUCCESS');
        const hasWarningJob = methodData.jobs && methodData.jobs.some(job => job.status === 'WARNING');
        const hasFailedJob = methodData.jobs && methodData.jobs.some(job => job.status === 'FAILED');
        const hasCanceledJob = methodData.jobs && methodData.jobs.some(job => job.status === 'CANCELED' || job.status === 'CANCELLED');

        // Extract metrics from the most recent job with results
        const latestJobWithResults = methodData.jobs && methodData.jobs.length > 0 
            ? methodData.jobs.slice().reverse().find(job => job.result && job.result.summary) 
            : null;

        // Prepare metrics cards data
        const metricsCards = [];
        if (latestJobWithResults && latestJobWithResults.result.summary) {
            const summary = latestJobWithResults.result.summary;
            Object.entries(summary).forEach(([metricName, metricData]) => {
                if (metricData && typeof metricData.mean !== 'undefined') {
                    metricsCards.push({
                        name: metricName.toUpperCase(),
                        value: Number(metricData.mean).toFixed(2),
                        icon: 'fa-chart-line',
                        color: 'text-green-600'
                    });
                }
            });
        }

        // Define all possible cards in priority order
        const allPossibleCards = [
            ...metricsCards, // Metrics from latest job come first
            {
                name: 'Dataset',
                value: methodData.dataset_name,
                icon: 'fa-database',
                color: 'text-blue-500'
            },
            {
                name: 'Jobs',
                value: methodData.jobs ? methodData.jobs.length : 0,
                icon: 'fa-tasks',
                color: 'text-orange-500'
            },
            {
                name: 'Status',
                value: hasSuccessfulJob ? 'Completed' : hasWarningJob ? 'Warning' : hasFailedJob ? 'Failed' : hasCanceledJob ? 'Canceled' : 'Not Run',
                icon: 'fa-info-circle',
                color: hasSuccessfulJob ? 'text-green-600' : hasWarningJob ? 'text-orange-600' : hasFailedJob ? 'text-red-600' : hasCanceledJob ? 'text-gray-500' : 'text-gray-500'
            }
        ];

        // Take the first 4 cards, or fill with "No Data" cards if needed
        const displayCards = allPossibleCards.slice(0, 4);
        while (displayCards.length < 4) {
            displayCards.push({
                name: 'No Data',
                value: '—',
                icon: 'fa-minus',
                color: 'text-gray-400'
            });
        }

        const rendered = this.renderTemplate(template, {
            projectId: testData.project_id,
            testId: testData.id,
            testName: testData.name,
            methodId: methodData.id,
            methodIds: JSON.stringify([methodData.id]),
            methodName: methodData.name,
            methodDescription: methodData.description || 'No description available',
            datasetName: methodData.dataset_name,
            metrics: methodData.metrics.join(', '),
            hasResults: methodData.result_ids && methodData.result_ids.length > 0,
            jobCount: methodData.jobs ? methodData.jobs.length : 0,
            resultCount: methodData.result_ids ? methodData.result_ids.length : 0,
            parameters: JSON.stringify(methodData.parameters, null, 2),
            lastJobStatus: latestJob ? latestJob.status : 'No jobs',
            lastJobMessage: latestJob ? latestJob.status_message || '' : '',
            statusClass: hasSuccessfulJob ? 'text-green-600' : hasWarningJob ? 'text-orange-600' : hasFailedJob ? 'text-red-600' : hasCanceledJob ? 'text-gray-500' : 'text-gray-500',
            statusText: hasSuccessfulJob ? 'Completed' : hasWarningJob ? 'Warning' : hasFailedJob ? 'Failed' : hasCanceledJob ? 'Canceled' : 'Not Run',
            cards: displayCards,
            jobType: testData.diagnostic_only ? 'DIAGNOSIS' : 'FULL',
            jobButtonText: testData.diagnostic_only ? 'Run Diagnostics' : 'Run Complete Job',
        });

        this.switchToContent(rendered);
        
        // Add results section after rendering
        await this.addResultsSection(methodData);
        
        this.currentContent = { type: 'method', id: methodId, testId: testId };
        
        // Highlight the selected method
        this.highlightMethod(methodId);
    }

    async addResultsSection(methodData) {
        // Initialize tab functionality
        this.initializeTabs(methodData);
        
        // Populate jobs tab with job groups
        await this.populateJobsTab(methodData);
        
        // Populate activity history
        this.populateActivityHistory(methodData);
    }

    initializeTabs(methodData) {
        // Add event listeners for tab switching
        const tabButtons = document.querySelectorAll('.method-tab-btn');
        const tabContents = document.querySelectorAll('.method-tab-content');
        const compareBtn = document.getElementById('compare-jobs-btn');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');
                
                // Update button states
                tabButtons.forEach(btn => {
                    btn.classList.remove('active', 'border-primary', 'text-primary');
                    btn.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
                });
                button.classList.add('active', 'border-primary', 'text-primary');
                button.classList.remove('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
                
                // Show/hide tab content
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                });
                document.getElementById(`${targetTab}-tab`).classList.remove('hidden');
                
                // Show/hide compare button based on active tab
                if (targetTab === 'jobs') {
                    compareBtn.style.display = 'block';
                } else {
                    compareBtn.style.display = 'none';
                }
            });
        });
        
        // Set initial active state
        const firstTab = tabButtons[0];
        if (firstTab) {
            firstTab.classList.add('border-primary', 'text-primary');
            firstTab.classList.remove('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
        }
    }

    async populateJobsTab(methodData) {
        const hasJobs = methodData.jobs && methodData.jobs.length > 0;
        const jobGroupsContainer = document.getElementById('job-groups-container');
        const compareBtn = document.getElementById('compare-jobs-btn');
        
        if (!hasJobs) {
            jobGroupsContainer.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-tasks text-4xl text-gray-300 mb-4"></i>
                    <h3 class="text-lg font-medium text-gray-600 mb-2">No Jobs Yet</h3>
                    <p class="text-gray-500">Job groups will appear here after running this method.</p>
                </div>
            `;
            return;
        }

        // Save current scroll position
        const scrollPosition = jobGroupsContainer.scrollTop;
        
        // Save current expanded/collapsed states before regenerating
        const expandButtons = jobGroupsContainer.querySelectorAll('[data-param-hash]');
        expandButtons.forEach(button => {
            const paramHash = button.getAttribute('data-param-hash');
            const isExpanded = button.getAttribute('data-expanded') === 'true';
            this.jobGroupStates[paramHash] = isExpanded;
        });

        // Load tag data for all jobs before displaying them
        await this.loadTagsForJobs(methodData.jobs);

        // Check for matching tags from search
        const matchingTags = window.currentMatchingTags || [];
        let matchingJobsCount = 0;
        
        // If we have matching tags, count how many jobs match
        if (matchingTags.length > 0) {
            matchingJobsCount = methodData.jobs.filter(job => {
                if (!job.tags || job.tags.length === 0) return false;
                return job.tags.some(tag => {
                    const tagName = (tag.name || tag || '').toLowerCase();
                    return matchingTags.some(searchTag => 
                        tagName.includes(searchTag.toLowerCase())
                    );
                });
            }).length;
        }

        // Group jobs by parameters
        const jobGroups = this.groupJobsByParameters(methodData.jobs);
        const jobGroupsSize = jobGroups.length;

        let groupsHtml = '';
        jobGroups.forEach((group, groupIndex) => {
            const paramHash = JSON.stringify(group.parameters);
            const shouldBeExpanded = this.jobGroupStates.hasOwnProperty(paramHash) 
                ? this.jobGroupStates[paramHash] 
                : true;
            
            const successfulJobs = group.jobs.filter(job => job.status === 'COMPLETED' || job.status === 'SUCCESS').length;
            const totalJobs = group.jobs.length;
            const successRate = totalJobs > 0 ? ((successfulJobs / totalJobs) * 100).toFixed(1) : 0;
            
            // Calculate average runtime (fake data for now)
            const avgRuntime = this.calculateAverageRuntime(group.jobs);
            
            // Count how many jobs in this group have matching tags
            let groupTaggedCount = 0;
            if (matchingTags.length > 0) {
                groupTaggedCount = group.jobs.filter(job => {
                    if (!job.tags || job.tags.length === 0) return false;
                    return job.tags.some(tag => {
                        const tagName = (tag.name || tag || '').toLowerCase();
                        return matchingTags.some(searchTag => 
                            tagName.includes(searchTag.toLowerCase())
                        );
                    });
                }).length;
            }
            
            groupsHtml += `
                <div class="mb-4 border border-gray-200 rounded-lg overflow-hidden">
                    <div class="bg-gray-50 px-4 py-3 border-b border-gray-200">
                        <div class="flex items-center justify-between">
                            <button class="flex items-center space-x-3 text-left flex-grow hover:text-primary transition-colors group-toggle" 
                                    onclick="toggleJobGroup(this)" data-expanded="${shouldBeExpanded}" data-param-hash="${this.escapeHtml(paramHash)}">
                                <i class="fas fa-chevron-${shouldBeExpanded ? 'down' : 'right'} text-sm text-gray-400 transition-transform"></i>
                                <div class="flex-1">
                                    <div class="flex items-center justify-between">
                                        <h4 class="font-medium text-gray-800">Parameter Set ${jobGroupsSize - groupIndex}</h4>
                                        <div class="flex items-center space-x-4 text-sm text-gray-500">
                                            <span class="flex items-center">
                                                <i class="fas fa-tasks mr-1"></i>
                                                ${totalJobs} job${totalJobs > 1 ? 's' : ''}
                                            </span>
                                            <span class="flex items-center">
                                                <i class="fas fa-clock mr-1"></i>
                                                ${avgRuntime}
                                            </span>
                                            <span class="flex items-center">
                                                <i class="fas fa-check-circle mr-1 ${successRate > 80 ? 'text-green-500' : successRate > 50 ? 'text-yellow-500' : 'text-red-500'}"></i>
                                                ${successRate}% success
                                            </span>
                                            ${groupTaggedCount > 0 ? `
                                            <span class="flex items-center text-blue-600 font-medium">
                                                <i class="fas fa-tag mr-1"></i>
                                                ${groupTaggedCount} tagged
                                            </span>
                                            ` : ''}
                                        </div>
                                    </div>
                                </div>
                            </button>
                            <button class="ml-3 px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors"
                                    onclick="openParametersModal(${groupIndex}, event)"
                                    title="View Parameters">
                                <i class="fas fa-cog mr-1"></i>
                                Parameters
                            </button>
                        </div>
                    </div>
                    <div class="job-group-content ${shouldBeExpanded ? '' : 'hidden'} p-4">
                        <div class="space-y-3 max-h-80 overflow-y-auto">
            `;
            
            group.jobs.forEach((job, jobIndex) => {
                const statusColor = job.status === 'COMPLETED' || job.status === 'SUCCESS' ? 'text-green-600' : job.status === 'WARNING' ? 'text-orange-600' : job.status === 'FAILED' ? 'text-red-600' : job.status === 'CANCELED' || job.status === 'CANCELLED' ? 'text-gray-500' : job.status === 'PROCESSING' ? 'text-blue-600' : 'text-yellow-600';
                const statusIcon = job.status === 'COMPLETED' || job.status === 'SUCCESS' ? 'fa-check-circle' : job.status === 'WARNING' ? 'fa-exclamation-triangle' : job.status === 'FAILED' ? 'fa-times-circle' : job.status === 'CANCELED' || job.status === 'CANCELLED' ? 'fa-ban' : 'fa-clock';
                const runtime = this.formatRuntime(job.runtime || 0);
                const timestamp = this.formatTimestamp(job.created_at || new Date(Date.now()));
                
                // Check if this job matches any of the search tags
                let hasMatchingTag = false;
                if (matchingTags.length > 0 && job.tags && job.tags.length > 0) {
                    hasMatchingTag = job.tags.some(tag => {
                        const tagName = (tag.name || tag || '').toLowerCase();
                        return matchingTags.some(searchTag => 
                            tagName.includes(searchTag.toLowerCase())
                        );
                    });
                }
                
                // Enhanced tag indicator with both count and preview of tag names
                let tagIndicator = '';
                if (job.tags && job.tags.length > 0) {
                    const tagCount = job.tags.length;
                    const tagNames = job.tags.slice(0, 2).map(tag => tag.name || tag).join(', ');
                    const allTagNames = job.tags.map(tag => tag.name || tag).join(', ');
                    const moreText = tagCount > 2 ? ` +${tagCount - 2}` : '';
                    
                    tagIndicator = `
                        <div class="flex items-center space-x-1 text-xs bg-blue-50 border border-blue-200 px-2 py-1 rounded-full max-w-xs" title="${allTagNames}">
                            <i class="fas fa-tag text-blue-600 flex-shrink-0"></i>
                            <span class="text-blue-700 truncate">${tagNames}${moreText}</span>
                        </div>
                    `;
                }
                
                // Check if job has new logs from queue manager
                const queueJob = window.queueManager ? window.queueManager.jobs.find(j => j.id === job.id) : null;
                const newLogsIndicator = (queueJob && queueJob.hasNewLogs) ? 
                    '<span class="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse ml-1" title="New logs available"></span>' : '';
                
                groupsHtml += `
                    <div class="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 job-card-clickable transition-colors duration-200 ${hasMatchingTag ? 'job-card-with-tag-highlight border-2 border-blue-500' : ''}" onclick="openJobDetailsModal('${job.id}', event)" data-job-id="${job.id}">
                        <label class="custom-checkbox" onclick="event.stopPropagation()">
                            <input type="checkbox" class="job-checkbox" data-job-id="${job.id}" onchange="updateCompareButton()">
                            <span class="checkbox-indicator"></span>
                        </label>
                        <div class="flex items-center space-x-3 flex-grow pointer-events-none">
                            <i class="fas ${statusIcon} ${statusColor}"></i>
                            <div class="flex-grow">
                                <div class="flex items-center justify-between h-full">
                                    <div class="flex flex-col justify-center min-w-0 flex-1">
                                        <div class="flex items-center space-x-2">
                                            <span class="font-medium text-gray-800">Job #${job.id}${newLogsIndicator}</span>
                                            <div class="pointer-events-auto">${tagIndicator}</div>
                                        </div>
                                        <div class="text-sm text-gray-500 mt-1 truncate max-w-xl">${job.status_message}</div>
                                    </div>
                                    <div class="flex flex-col items-end justify-center">
                                        <div class="flex items-center space-x-4 text-sm">
                                            <span class="text-gray-500">${runtime}</span>
                                            <span class="font-medium ${statusColor}">${job.status}</span>
                                        </div>
                                        <div class="text-xs text-gray-400">${timestamp}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="flex-shrink-0 pointer-events-none">
                            <i class="fas fa-chevron-right text-gray-400 text-sm"></i>
                        </div>
                    </div>
                `;
            });
            groupsHtml += '</div></div></div>';
        });
        
        jobGroupsContainer.innerHTML = groupsHtml;
        
        // Add count indicator if there are matching jobs
        if (matchingJobsCount > 0) {
            const tabHeader = document.querySelector('.method-tab-btn[data-tab="jobs"]');
            if (tabHeader && !document.getElementById('tag-match-count')) {
                const countBadge = document.createElement('span');
                countBadge.id = 'tag-match-count';
                countBadge.className = 'ml-2 px-2 py-0.5 text-xs bg-blue-500 text-white rounded-full';
                countBadge.textContent = `${matchingJobsCount} tagged`;
                countBadge.title = `${matchingJobsCount} job${matchingJobsCount > 1 ? 's' : ''} match${matchingJobsCount === 1 ? 'es' : ''} tags: ${matchingTags.join(', ')}`;
                tabHeader.appendChild(countBadge);
            }
        }
        
        // Restore scroll position
        jobGroupsContainer.scrollTop = scrollPosition;
        
        // Store job groups data for modal access
        this.currentJobGroups = jobGroups;
        
        // Set up compare functionality
        window.updateCompareButton = () => {
            const selectedJobs = document.querySelectorAll('.job-checkbox:checked');
            compareBtn.disabled = selectedJobs.length === 0 || selectedJobs.length > 2;
        };
        
        // Global function for toggling job selection when clicking the card
        window.toggleJobSelection = (jobId, event) => {
            if (event) {
                event.preventDefault();
            }
            const checkbox = document.querySelector(`.job-checkbox[data-job-id="${jobId}"]`);
            if (checkbox) {
                checkbox.checked = !checkbox.checked;
                updateCompareButton();
            }
        };
        
        compareBtn.addEventListener('click', () => {
            const selectedJobs = document.querySelectorAll('.job-checkbox:checked');
            const jobIds = Array.from(selectedJobs).map(checkbox => checkbox.getAttribute('data-job-id'));
            
            if (jobIds.length === 0 || jobIds.length > 2) {
                alert('Please select 1 or 2 jobs to view the report.');
                return;
            }
            
            // Get the current method and project IDs
            const methodId = this.currentContent.id;
            
            if (jobIds.length === 1) {
                // Single job - view individual report
                const compareUrl = `/projects/${projectId}/methods/${methodId}/compare/?jobs=${encodeURIComponent(jobIds[0])}`;
                window.location.href = compareUrl;
            } else {
                // Two jobs - view comparison report
                const queryParams = jobIds.map(id => `jobs=${encodeURIComponent(id)}`).join('&');
                const compareUrl = `/projects/${projectId}/methods/${methodId}/compare/?${queryParams}`;
                window.location.href = compareUrl;
            }
        });
        
        // Global function for toggling job groups
        window.toggleJobGroup = (button) => {
            const isExpanded = button.getAttribute('data-expanded') === 'true';
            const chevron = button.querySelector('i');
            const content = button.closest('.mb-4').querySelector('.job-group-content');
            const paramHash = button.getAttribute('data-param-hash');
            
            if (isExpanded) {
                // Collapse
                content.classList.add('hidden');
                chevron.classList.remove('fa-chevron-down');
                chevron.classList.add('fa-chevron-right');
                button.setAttribute('data-expanded', 'false');
                if (paramHash) contentManager.jobGroupStates[paramHash] = false;
            } else {
                // Expand
                content.classList.remove('hidden');
                chevron.classList.remove('fa-chevron-right');
                chevron.classList.add('fa-chevron-down');
                button.setAttribute('data-expanded', 'true');
                if (paramHash) contentManager.jobGroupStates[paramHash] = true;
            }
        };
        
        // Global function for opening parameters modal
        window.openParametersModal = (groupIndex, event) => {
            event.stopPropagation(); // Prevent triggering the toggle function
            
            const group = this.currentJobGroups[groupIndex];
            if (!group) return;
            
            const modal = document.getElementById('parameters-modal');
            const modalTitle = document.getElementById('modal-title');
            const modalParameters = document.getElementById('modal-parameters');
            const modalStatistics = document.getElementById('modal-statistics');
            
            // Update modal content
            const jobGroupsSize = this.currentJobGroups.length;
            modalTitle.textContent = `Parameter Set ${jobGroupsSize - groupIndex} Details`;
            modalParameters.textContent = JSON.stringify(group.parameters, null, 2);
            
            // Calculate and display statistics
            const totalJobs = group.jobs.length;
            const successfulJobs = group.jobs.filter(job => job.status === 'COMPLETED' || job.status === 'SUCCESS').length;
            const failedJobs = group.jobs.filter(job => job.status === 'FAILED').length;
            
            modalStatistics.innerHTML = `
                <div class="bg-blue-50 p-3 rounded-lg text-center">
                    <div class="text-xl font-bold text-blue-600">${totalJobs}</div>
                    <div class="text-sm text-blue-700">Total Jobs</div>
                </div>
                <div class="bg-green-50 p-3 rounded-lg text-center">
                    <div class="text-xl font-bold text-green-600">${successfulJobs}</div>
                    <div class="text-sm text-green-700">Successful</div>
                </div>
                <div class="bg-red-50 p-3 rounded-lg text-center">
                    <div class="text-xl font-bold text-red-600">${failedJobs}</div>
                    <div class="text-sm text-red-700">Failed</div>
                </div>
            `;
            
            // Show modal
            modal.classList.remove('hidden');
            
            // Add escape key listener
            const escapeHandler = (e) => {
                if (e.key === 'Escape') {
                    closeParametersModal();
                    document.removeEventListener('keydown', escapeHandler);
                }
            };
            document.addEventListener('keydown', escapeHandler);
        };
        
        // Global function for opening job details modal
        window.openJobDetailsModal = async (jobId, event) => {
            if (event) {
                event.stopPropagation();
            }
            
            // Find the job data and the test it belongs to
            const job = this.findJobById(jobId);
            const test = this.findTestByJobId(jobId);
            if (!job) {
                console.error(`Job with ID ${jobId} not found`);
                return;
            }
            
            const modal = document.getElementById('job-details-modal');
            const modalTitle = document.getElementById('job-modal-title');
            const modalId = document.getElementById('job-modal-id');
            const modalStatus = document.getElementById('job-modal-status');
            const modalStatusIcon = document.getElementById('job-modal-status-icon');
            const modalRuntime = document.getElementById('job-modal-runtime');
            const modalTimestamp = document.getElementById('job-modal-timestamp');
            const modalStatusMessage = document.getElementById('job-modal-status-message');
            
            // Update modal header
            modalTitle.textContent = `Job #${jobId} Details`;
            modalId.textContent = `Job #${jobId}`;
            
            // Update status
            const statusColor = job.status === 'COMPLETED' || job.status === 'SUCCESS' ? 'text-green-600' : job.status === 'WARNING' ? 'text-orange-600' : job.status === 'FAILED' ? 'text-red-600' : job.status === 'CANCELED' || job.status === 'CANCELLED' ? 'text-gray-500' : job.status === 'PROCESSING' ? 'text-blue-600' : 'text-yellow-600';
            const statusIcon = job.status === 'COMPLETED' || job.status === 'SUCCESS' ? 'fa-check-circle' : job.status === 'WARNING' ? 'fa-exclamation-triangle' : job.status === 'FAILED' ? 'fa-times-circle' : job.status === 'CANCELED' || job.status === 'CANCELLED' ? 'fa-ban' : 'fa-clock';
            
            modalStatus.textContent = job.status;
            modalStatus.className = `font-medium ${statusColor}`;
            modalStatusIcon.className = `fas ${statusIcon} ${statusColor}`;
            
            // Update runtime and timestamp
            const runtime = this.formatRuntime(job.runtime || 0);
            const timestamp = this.formatTimestamp(job.created_at || new Date(Date.now()));
            modalRuntime.textContent = runtime;
            modalTimestamp.textContent = timestamp;
            
            // Update status message
            if (job.status_message) {
                modalStatusMessage.textContent = job.status_message;
                modalStatusMessage.classList.remove('hidden');
            } else {
                modalStatusMessage.classList.add('hidden');
            }
            
            // Initialize tab functionality with test data
            this.initializeJobDetailTabs(test);
            
            // Populate content with mock data
            await this.populateJobModalContent(job);
            
            // Show modal
            modal.classList.remove('hidden');
            
            // Add escape key listener
            const escapeHandler = (e) => {
                if (e.key === 'Escape') {
                    closeJobDetailsModal();
                    document.removeEventListener('keydown', escapeHandler);
                }
            };
            document.addEventListener('keydown', escapeHandler);
        };
        
        // Global function for closing job details modal
        window.closeJobDetailsModal = () => {
            const modal = document.getElementById('job-details-modal');
            modal.classList.add('hidden');
        };
        
        // Global function for closing parameters modal
        window.closeParametersModal = () => {
            const modal = document.getElementById('parameters-modal');
            modal.classList.add('hidden');
        };
        
        // Global function for exporting results
        window.exportResults = (format = 'json') => {
            // Get current context from the content manager
            if (!window.contentManager || !window.contentManager.currentJob || !window.contentManager.currentContent) {
                console.error('No job or content context available for results export');
                return;
            }
            
            const currentJob = window.contentManager.currentJob;
            const currentContent = window.contentManager.currentContent;
            
            // Get method ID and job ID
            const methodId = currentContent.id;
            const jobIds = [currentJob.id];
            
            // Build the export URL based on format
            const baseUrl = baseUrlGenerateReport + `/methods/${methodId}/export-results/`;
            const exportUrl = `${baseUrl}?format=${format}&${jobIds.map(id => `jobs=${id}`).join('&')}`;

            // Trigger download
            window.open(exportUrl, '_blank');
        };

        // Global function for toggling detailed results
        window.toggleDetailedResults = () => {
            const content = document.getElementById('detailed-results-content');
            const chevron = document.getElementById('detailed-results-chevron');
            
            if (content.classList.contains('hidden')) {
                // Show content
                content.classList.remove('hidden');
                chevron.classList.remove('fa-chevron-right');
                chevron.classList.add('fa-chevron-down');
            } else {
                // Hide content
                content.classList.add('hidden');
                chevron.classList.remove('fa-chevron-down');
                chevron.classList.add('fa-chevron-right');
            }
        };
        
        // Close modal when clicking outside of it
        document.getElementById('parameters-modal').addEventListener('click', (e) => {
            if (e.target.id === 'parameters-modal') {
                closeParametersModal();
            }
        });


        
    }

    calculateAverageRuntime(jobs) {
        const completedJobs = jobs.filter(job => job.status === 'COMPLETED' || job.status === 'SUCCESS');
        if (completedJobs.length === 0) return '—';
        
        // For now, generate fake but consistent runtime data
        const totalRuntime = completedJobs.reduce((sum, job) => {
            return sum + (job.runtime || 0);
        }, 0);
        
        const avgSeconds = Math.floor(totalRuntime / completedJobs.length);
        return this.formatRuntime(avgSeconds);
    }

    formatRuntime(seconds) {
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    }

    formatTimestamp(date) {
        if (typeof date === 'string') date = new Date(date);
        const now = new Date();
        
        // Check if the date is today
        const isToday = date.getDate() === now.getDate() &&
                       date.getMonth() === now.getMonth() &&
                       date.getFullYear() === now.getFullYear();
        
        if (isToday) {
            // Show exact time for today (e.g., "2:30 PM")
            return date.toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
            });
        } else {
            // Show just the date for other days (e.g., "Dec 11, 2024")
            return date.toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric', 
                year: 'numeric' 
            });
        }
    }

    findJobById(jobId) {
        // Search through all tests and methods to find the job
        for (const test of this.testsData) {
            for (const method of test.methods) {
                if (method.jobs) {
                    const job = method.jobs.find(j => j.id == jobId);
                    if (job) {
                        return job;
                    }
                }
            }
        }
        return null;
    }

    findTestByJobId(jobId) {
        // Search through all tests and methods to find the test that contains the job
        for (const test of this.testsData) {
            for (const method of test.methods) {
                if (method.jobs) {
                    const job = method.jobs.find(j => j.id == jobId);
                    if (job) {
                        return test;
                    }
                }
            }
        }
        return null;
    }

    initializeJobDetailTabs(test = null) {
        const tabButtons = document.querySelectorAll('.job-detail-tab-btn');
        const tabContents = document.querySelectorAll('.job-detail-tab-content');
        const sampleNavigation = document.getElementById('sample-navigation');
        
        // Find the diagnosis tab button and always show it (diagnostics will be available when results contain them)
        const diagnosisTabButton = document.querySelector('.job-detail-tab-btn[data-tab="diagnosis"]');
        if (diagnosisTabButton) {
            diagnosisTabButton.style.display = '';
        }
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');
                
                // Update button states
                tabButtons.forEach(btn => {
                    btn.classList.remove('active', 'border-primary', 'text-primary');
                    btn.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
                });
                button.classList.add('active', 'border-primary', 'text-primary');
                button.classList.remove('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
                
                // Show/hide tab content
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                });
                document.getElementById(`${targetTab}-tab`).classList.remove('hidden');
                
                // Show/hide sample navigation and LXT buttons based on active tab
                if (sampleNavigation) {
                    if (targetTab === 'inference') {
                        sampleNavigation.classList.remove('hidden');
                    } else {
                        sampleNavigation.classList.add('hidden');
                    }
                }
                
                // Show/hide Diagnostic buttons based on active tab
                const diagnosticButtonsContainer = document.getElementById('diagnostic-buttons-container');
                if (diagnosticButtonsContainer) {
                    if (targetTab === 'inference') {
                        // Get task type from job parameters
                        const taskType = currentJobForDiagnostic && currentJobForDiagnostic.parameters && currentJobForDiagnostic.parameters["task_type"];
                        
                        // Show for completed/warning jobs with inference results, but not for text-retrieval tasks
                        if (currentJobForDiagnostic && (currentJobForDiagnostic.status === 'COMPLETED' || currentJobForDiagnostic.status === 'SUCCESS' || currentJobForDiagnostic.status === 'WARNING') && 
                            currentJobForDiagnostic.result && currentJobForDiagnostic.result.results && 
                            taskType !== 'text-retrieval') {
                            diagnosticButtonsContainer.style.display = 'flex';
                        } else {
                            diagnosticButtonsContainer.style.display = 'none';
                        }
                    } else {
                        diagnosticButtonsContainer.style.display = 'none';
                    }
                }
                
                // Clear new logs indicator when switching to logs tab
                if (targetTab === 'logs' && this.currentJobInModal) {
                    this.clearNewLogsIndicator(this.currentJobInModal);
                }
            });
        });
        
        // Set initial state based on which tab is actually active
        if (sampleNavigation) {
            const activeTab = document.querySelector('.job-detail-tab-btn.active');
            if (activeTab && activeTab.getAttribute('data-tab') === 'inference') {
                sampleNavigation.classList.remove('hidden');
            } else {
                sampleNavigation.classList.add('hidden');
            }
        }
        
        // Set initial state for Diagnostic buttons
        const diagnosticButtonsContainer = document.getElementById('diagnostic-buttons-container');
        if (diagnosticButtonsContainer) {
            const activeTab = document.querySelector('.job-detail-tab-btn.active');
            if (activeTab && activeTab.getAttribute('data-tab') === 'inference') {
                // Get task type from job parameters
                const taskType = currentJobForDiagnostic && currentJobForDiagnostic.parameters && currentJobForDiagnostic.parameters["task_type"];
                
                // Show for completed/warning jobs with inference results, but not for text-retrieval tasks
                if (currentJobForDiagnostic && (currentJobForDiagnostic.status === 'COMPLETED' || currentJobForDiagnostic.status === 'SUCCESS' || currentJobForDiagnostic.status === 'WARNING') && 
                    currentJobForDiagnostic.result && currentJobForDiagnostic.result.results && 
                    taskType !== 'text-retrieval') {
                    diagnosticButtonsContainer.style.display = 'flex';
                } else {
                    diagnosticButtonsContainer.style.display = 'none';
                }
            } else {
                diagnosticButtonsContainer.style.display = 'none';
            }
        }
    }

    async populateJobModalContent(job) {
        // Track the current job being viewed in the modal
        this.currentJobInModal = job.id;
        
        // Set global variables for diagnostic functionality
        currentJobForDiagnostic = job;
        currentSampleForDiagnostic = null; // Will be updated when samples are navigated
        
        // Show/hide Diagnostic buttons based on job completion status and task type
        const diagnosticButtonsContainer = document.getElementById('diagnostic-buttons-container');
        if (diagnosticButtonsContainer) {
            // Get task type from job parameters
            const taskType = job.parameters && job.parameters["task_type"];
            
            // Show Diagnostic buttons for completed/warning jobs with inference results, but not for text-retrieval tasks
            if ((job.status === 'COMPLETED' || job.status === 'SUCCESS' || job.status === 'WARNING') && 
                job.result && job.result.results && 
                taskType !== 'text-retrieval') {
                diagnosticButtonsContainer.style.display = 'flex';
            } else {
                diagnosticButtonsContainer.style.display = 'none';
            }
        }
        
        // Update button states based on data availability
        this.updateDiagnosticButtonStates();
        
        // Populate inference tab with mock data
        this.populateInferenceTab(job);
        
        // Populate analysis tab with mock data
        this.populateAnalysisTab(job);
        
        // Populate diagnosis tab with mock data
        this.populateDiagnosisTab(job);
        
        // Populate tags tab (await to prevent blocking)
        await this.populateTagsTab(job);
        
        // Populate logs tab with mock data
        this.populateLogsTab(job);
    }

    updateDiagnosticButtonStates() {
        const diagnosticButton = document.getElementById('run-diagnostic-btn');
        const diagnosticOptions = document.getElementById('run-diagnostic-options');
        const lxtSampleOption = document.getElementById('diagnostic-lxt-sample-option');
        const shapSampleOption = document.getElementById('diagnostic-shap-sample-option');
        
        if (diagnosticButton) {
            // Main button is enabled if there's a valid job with completed/warning status
            diagnosticButton.disabled = !currentJobForDiagnostic || (currentJobForDiagnostic.status !== 'COMPLETED' && currentJobForDiagnostic.status !== 'SUCCESS' && currentJobForDiagnostic.status !== 'WARNING');
        }
        
        if (diagnosticOptions) {
            // Options button follows main button state
            diagnosticOptions.disabled = !currentJobForDiagnostic || (currentJobForDiagnostic.status !== 'COMPLETED' && currentJobForDiagnostic.status !== 'SUCCESS' && currentJobForDiagnostic.status !== 'WARNING');
        }
        
        // Sample options are disabled if no current sample
        if (lxtSampleOption) {
            if (!currentSampleForDiagnostic) {
                lxtSampleOption.classList.add('opacity-50', 'cursor-not-allowed');
                lxtSampleOption.disabled = true;
            } else {
                lxtSampleOption.classList.remove('opacity-50', 'cursor-not-allowed');
                lxtSampleOption.disabled = false;
            }
        }
        
        if (shapSampleOption) {
            if (!currentSampleForDiagnostic) {
                shapSampleOption.classList.add('opacity-50', 'cursor-not-allowed');
                shapSampleOption.disabled = true;
            } else {
                shapSampleOption.classList.remove('opacity-50', 'cursor-not-allowed');
                shapSampleOption.disabled = false;
            }
        }
    }

    populateInferenceTab(job) {
        // Store job data for navigation
        this.currentJob = job;
        this.currentSampleIndex = 0;
        this.totalSamples = (job.result && job.result.results) ? job.result.results.length : 0;
        
        // Extract column names from parameters
        this.currentInputColumn = job.parameters?.dataset?.input_column || 'input';
        this.currentReferenceColumn = job.parameters?.dataset?.reference_column || 'reference';
        
        // Update the sample counter and navigation buttons
        this.updateSampleNavigation();
        
        // Display the current sample
        this.displayCurrentSample();
    }
    
    updateSampleNavigation() {
        const sampleCounter = document.getElementById('sample-counter');
        const prevButton = document.getElementById('prev-sample-btn');
        const nextButton = document.getElementById('next-sample-btn');
        
        if (!sampleCounter || !prevButton || !nextButton) return;
        
        if (this.totalSamples > 0) {
            sampleCounter.textContent = `${this.currentSampleIndex + 1} of ${this.totalSamples}`;
            
            // Enable/disable navigation buttons
            prevButton.disabled = this.currentSampleIndex === 0;
            nextButton.disabled = this.currentSampleIndex === this.totalSamples - 1;
        } else {
            sampleCounter.textContent = '0 of 0';
            prevButton.disabled = true;
            nextButton.disabled = true;
        }
    }
    
    displayCurrentSample() {
        const inputContent = document.getElementById('job-input-content');
        const documentsTableContainer = document.getElementById('documents-table-container');
        const simpleResultsContainer = document.getElementById('simple-results-container');
        const noResultsContainer = document.getElementById('no-inference-results');
        const noResultsMessage = document.getElementById('no-results-message');
        if (!inputContent) return;
        
        if (this.currentJob.result && this.currentJob.result.results && this.currentJob.result.results.length > 0) {
            // Hide no results message
            noResultsContainer.classList.add('hidden');
            
            // Get the current sample
            const currentResult = this.currentJob.result.results[this.currentSampleIndex];
            
            // Update the global current sample for diagnostics
            currentSampleForDiagnostic = {
                input: currentResult.input || currentResult.query || '',
                reference: currentResult.reference || currentResult.target || '',
                sample_index: this.currentSampleIndex
            };
            
            // Update diagnostic button states
            this.updateDiagnosticButtonStates();
            
            // Determine task type from job data
            const taskType = this.currentJob.parameters["task_type"];
            
            // Update input
            this.updateInputContent(currentResult);
            
            // Update results based on task type
            if (taskType === 'text-retrieval') {
                documentsTableContainer.classList.remove('hidden');
                simpleResultsContainer.classList.add('hidden');
                this.updateDocumentsTable(currentResult);
            } else {
                documentsTableContainer.classList.add('hidden');
                simpleResultsContainer.classList.remove('hidden');
                this.updateSimpleResults(currentResult, taskType);
            }
        } else {
            // Hide result containers, show no results message
            documentsTableContainer.classList.add('hidden');
            simpleResultsContainer.classList.add('hidden');
            noResultsContainer.classList.remove('hidden');
            
            // Reset current sample for diagnostics
            currentSampleForDiagnostic = null;
            
            // Update diagnostic button states
            this.updateDiagnosticButtonStates();
            
            // Update no results message based on job status
            const message = this.currentJob.status === 'FAILED' 
                ? 'Job failed - no inference results available'
                : this.currentJob.status === 'COMPLETED' || this.currentJob.status === 'SUCCESS'
                    ? 'No detailed inference results available for this job'
                    : this.currentJob.status === 'PROCESSING' || this.currentJob.status === 'QUEUED'
                        ? 'Job is still processing - inference results will be available once the job is completed'
                        : 'No inference results available for this job';
            
            noResultsMessage.textContent = message;
            inputContent.textContent = message;
        }
    }

    updateInputContent(result) {
        const inputHeader = document.querySelector('#inference-tab .bg-white.shadow.rounded-lg h4');
        
        if (!result.input) {
            setInputContent('');
            return;
        }
        
        setInputContent(this.escapeHtml(result.input));
        
        // Update header with column name
        if (inputHeader && this.currentInputColumn) {
            const columnLabel = this.currentInputColumn !== 'input' ? ` ('${this.currentInputColumn}' column)` : '';
            inputHeader.innerHTML = `<i class="fas fa-arrow-right mr-2 text-blue-500"></i>Input${columnLabel}`;
        }
    }

    updateDocumentsTable(result) {
        const tableBody = document.getElementById('documents-table-body');
        if (!tableBody) return;
        
        if (!result.documents || result.documents.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="px-4 py-8 text-center text-gray-500">
                        No retrieved documents available
                    </td>
                </tr>
            `;
            return;
        }
        
        let html = '';
        result.documents.forEach((doc, index) => {
            const prediction = result.prediction && result.prediction[index] !== undefined 
                ? this.formatScore(result.prediction[index]) 
                : '—';
                
            const reference = result.reference && result.reference[index] !== undefined 
                ? result.reference[index] 
                : '—';
                
            const isRelevant = reference === 1 || reference === '1';
            
            html += `
                <tr class="hover:bg-gray-50 cursor-pointer doc-row" data-doc-index="${index}">
                    <td class="px-4 py-3 text-sm font-medium text-gray-900">
                        ${doc.doc_id}
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-700">
                        <div class="max-w-md">
                            ${this.truncateText(this.escapeHtml(doc.doc_text), 150)}
                            ${doc.doc_text.length > 150 ? ' <i class="fas fa-expand-arrows-alt ml-1 text-primary text-xs"></i>' : ''}
                        </div>
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-700 text-center">
                        ${doc.bm25_score !== undefined ? doc.bm25_score.toFixed(3) : '—'}
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-700 text-center">
                        ${doc.rerank_score !== undefined ? doc.rerank_score.toFixed(3) : '—'}
                    </td>
                    <td class="px-4 py-3 text-sm text-gray-700 text-center font-mono">
                        ${prediction}
                    </td>
                    <td class="px-4 py-3 text-sm text-center">
                        <span class="font-mono ${isRelevant ? 'text-green-600 font-semibold' : 'text-gray-400'}">${reference}</span>
                    </td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = html;
        
        // Attach click event listeners to each row
        tableBody.querySelectorAll('.doc-row').forEach((row, index) => {
            const doc = result.documents[index];
            row.addEventListener('click', () => {
                showDocumentModal(doc.doc_text, doc.doc_idx);
            });
        });
    }

    updateSimpleResults(result, taskType) {
        // Format output
        if (result.prediction) {
            let prediction = result.prediction;
            // Handle nested arrays
            while (Array.isArray(prediction) && prediction.length > 0) {
                prediction = prediction[0];
            }
            setOutputContent(prediction);
        } else {
            setOutputContent('');
        }
        
        // Format reference
        if (result.reference) {
            let reference = result.reference;
            // Handle nested arrays (e.g., for SARI metric: [[string]])
            while (Array.isArray(reference) && reference.length > 0) {
                reference = reference[0];
            }
            setReferenceContent(reference);
        } else {
            setReferenceContent('');
        }
        
        // Update reference header with column name
        const referenceHeader = document.querySelector('#simple-results-container > div:last-child h4');
        if (referenceHeader && this.currentReferenceColumn) {
            const columnLabel = this.currentReferenceColumn !== 'reference' ? ` ('${this.currentReferenceColumn}' column)` : '';
            referenceHeader.innerHTML = `<i class="fas fa-star mr-2 text-yellow-500"></i>Reference${columnLabel}`;
        }
    }

    formatScore(score) {
        if (typeof score === 'number') {
            return score.toFixed(2);
        }
        return score !== null && score !== undefined ? score.toString() : '—';
    }

    truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    navigateSample(direction) {
        if (!this.currentJob || this.totalSamples <= 1) return;
        
        if (direction === 'next' && this.currentSampleIndex < this.totalSamples - 1) {
            this.currentSampleIndex++;
        } else if (direction === 'prev' && this.currentSampleIndex > 0) {
            this.currentSampleIndex--;
        }
        
        // Reset LXT sample and output token indices when switching inference samples
        this.currentLXTSampleIndex = 0;
        this.currentOutputTokenIndex = 0;
        
        // Reset SHAP indices when switching samples
        this.currentSHAPSampleIndex = 0;
        this.lockedInputToken = null;
        
        // Update navigation and display
        this.updateSampleNavigation();
        this.displayCurrentSample();
    }

    getTaskLabel(taskType) {
        const labels = {
            'text-retrieval': 'Query',
            'summarization': 'Text',
            'text-simplification': 'Text'
        };
        return labels[taskType] || 'Query';
    }

    populateAnalysisTab(job) {
        const metricsGrid = document.getElementById('job-metrics-grid');
        const detailedResults = document.getElementById('job-detailed-results');
        const detailedResultsContainer = detailedResults?.closest('.bg-gray-50');
        
        // Show results if they exist, regardless of job status (including WARNING)
        if (job.result && job.result.summary && Object.keys(job.result.summary).length > 0) {
            // Show detailed results container when results are available
            if (detailedResultsContainer) {
                detailedResultsContainer.classList.remove('hidden');
            }
            const summary = job.result.summary;
            
            // Create metrics cards from actual data
            let metricsHtml = '';
            
            // Helper function to format metric names
            const formatMetricName = (name) => name.replace(/_/g, ' ')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            
            // Group metrics by source_metric for organized display
            const metricsBySource = {};
            Object.entries(summary).forEach(([metricName, metricData]) => {
                if (metricData && typeof metricData === 'object' && metricData.mean !== undefined) {
                    const source = metricData.source_metric || metricName;
                    if (!metricsBySource[source]) {
                        metricsBySource[source] = [];
                    }
                    metricsBySource[source].push({ name: metricName, data: metricData });
                }
            });
            
            // Process metrics grouped by source
            Object.entries(metricsBySource).forEach(([sourceMetric, metrics]) => {
                metrics.forEach(({ name: metricName, data: metricData }) => {
                    const displayName = formatMetricName(metricName);
                    const sourceFormatted = formatMetricName(sourceMetric);
                    
                    // Always show the breadcrumb with source metric tag
                    const metricLabel = `
                        <div class="flex items-center text-sm">
                            <span class="text-gray-500 font-normal text-xs" title="Source metric: ${sourceFormatted}">${sourceFormatted}</span>
                            <i class="fas fa-chevron-right mx-1 text-gray-400" style="font-size: 0.625rem;"></i>
                            <span class="font-medium text-gray-800" title="Metric: ${displayName}">${displayName}</span>
                        </div>
                    `;
                    
                    // Use consistent styling for all metrics
                    const color = 'text-primary';
                    const icon = 'fa-chart-line';
                    
                    // Calculate coefficient of variation (CV) as percentage for display
                    // CV = (std / mean) * 100
                    const stdValue = metricData.std ? metricData.std.toFixed(2) : '0.00';
                    const cvValue = metricData.std && metricData.mean ? 
                        ((metricData.std / Math.abs(metricData.mean)) * 100).toFixed(1) : '0.0';
                    
                    // Build the stats display based on available data
                    let statsDisplay = '';
                    if (metricData.min !== undefined && metricData.max !== undefined) {
                        statsDisplay = `Min: ${metricData.min.toFixed(2)} | Max: ${metricData.max.toFixed(2)}`;
                    } else if (metricData.count !== undefined) {
                        statsDisplay = `Count: ${metricData.count}`;
                    } else {
                        statsDisplay = 'Statistical data available';
                    }
                    
                    metricsHtml += `
                        <div class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                            <div class="flex items-center space-x-2 mb-2">
                                <i class="fas ${icon} ${color}"></i>
                                ${metricLabel}
                            </div>
                            <div class="flex items-baseline space-x-2 mb-1">
                                <span class="text-xs text-gray-500 font-medium">Mean:</span>
                                <div class="text-2xl font-bold ${color}" title="Mean Value: ${metricData.mean.toFixed(2)}">
                                    ${metricData.mean.toFixed(2)}
                                </div>
                            </div>
                            <div class="text-xs text-gray-500">
                                <div title="Range and additional statistics">${statsDisplay}</div>
                                <div class="mt-0.5" title="Standard Deviation: ${stdValue} | Coefficient of Variation: ${cvValue}% (relative to mean of ${metricData.mean.toFixed(2)})">Std: ${stdValue} (±${cvValue}%)</div>
                            </div>
                        </div>
                    `;
                });
            });
            
            // Fill remaining slots if needed
            const currentCards = (metricsHtml.match(/class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm"/g) || []).length;
            const maxCards = 6;
            
            for (let i = currentCards; i < maxCards; i++) {
                metricsHtml += `
                    <div class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm opacity-50">
                        <div class="flex items-center justify-between mb-2">
                            <div class="flex items-center space-x-2">
                                <i class="fas fa-minus text-gray-400"></i>
                                <span class="font-medium text-gray-400">No Data</span>
                            </div>
                        </div>
                        <div class="text-2xl font-bold text-gray-400">—</div>
                    </div>
                `;
            }
            
            metricsGrid.innerHTML = metricsHtml;
            
            // Show the full analysis results
            detailedResults.textContent = JSON.stringify(job.result.summary, null, 2);
        } else {
            // Hide detailed results container when no results are available
            if (detailedResultsContainer) {
                detailedResultsContainer.classList.add('hidden');
            }
            
            metricsGrid.innerHTML = `
                <div class="col-span-full text-center py-8">
                    <i class="fas fa-exclamation-triangle text-4xl text-gray-300 mb-4"></i>
                    <h3 class="text-lg font-medium text-gray-600 mb-2">No Analysis Results</h3>
                    <p class="text-gray-500">Analysis results will be available after successful job completion.</p>
                </div>
            `;
            detailedResults.textContent = job.status === 'FAILED' 
                ? 'Job failed - no analysis results available.' 
                : 'No analysis results available for this job.';
        }
    }

    populateDiagnosisTab(job) {
        const diagnosisContainer = document.getElementById('diagnosis-content');
        
        // Clear any previous diagnostic data to avoid rendering diagnostics from previous jobs
        this.currentLXTData = null;
        this.currentLXTSampleIndex = 0;
        this.totalLXTSamples = 0;
        this.currentOutputTokenIndex = 0;
        this.currentSHAPData = null;
        this.currentSHAPSampleIndex = 0;
        this.totalSHAPSamples = 0;
        this.lockedOutputToken = null;
        this.currentSHAPEncoderData = null;
        this.currentSHAPEncoderSampleIndex = 0;
        this.currentSHAPEncoderDocIndex = 0;
        this.totalSHAPEncoderSamples = 0;
        this.totalSHAPEncoderDocs = 0;
        
        if (job.result && job.result.diagnostics && job.status === 'COMPLETED' || job.status === 'SUCCESS') {
            const diagnostics = job.result.diagnostics;
            
            // Check if there are any diagnostics
            if (Object.keys(diagnostics).length === 0) {
                diagnosisContainer.innerHTML = `
                    <div class="text-center py-8">
                        <i class="fas fa-info-circle text-4xl text-gray-300 mb-4"></i>
                        <h3 class="text-lg font-medium text-gray-600 mb-2">No Diagnostics Available</h3>
                        <p class="text-gray-500">No diagnostic methods were run for this job.</p>
                    </div>
                `;
                return;
            }
            
            // Store diagnostic reports for expansion functionality
            this.currentDiagnosticReports = {};
            
            // Create content for each diagnostic method
            let diagnosticsHtml = '';
            Object.entries(diagnostics).forEach(([methodName, report], index) => {
                // Check if this is LXT format (array of objects where each object represents one output token)
                // LXT format: array of objects, each with a single key (output token) containing relevance scores
                const isLXTFormat = methodName === 'lxt' && Array.isArray(report) && report.length > 0 && 
                    typeof report[0] === 'object' && report[0] !== null &&
                    !Array.isArray(report[0]) && Object.keys(report[0]).length > 0;
                
                const isSHAPFormat = methodName === 'shap' && Array.isArray(report) && report.length > 0 &&
                    typeof report[0] === 'object' && report[0] !== null &&
                    report[0].attribution_matrix !== undefined;
                
                const isSHAPEncoderFormat = methodName === 'shap_encoder' && Array.isArray(report) && report.length > 0 &&
                    typeof report[0] === 'object' && report[0] !== null &&
                    report[0].attributions !== undefined && Array.isArray(report[0].attributions);
                
                const isIGEncoderFormat = methodName === 'integrated_gradients_encoder' && Array.isArray(report) && report.length > 0 &&
                    typeof report[0] === 'object' && report[0] !== null &&
                    report[0].attributions !== undefined && Array.isArray(report[0].attributions) &&
                    report[0].attributions[0]?.query_convergence_delta !== undefined;
                
                let reportContent;
                let needsExpansion = false;
                
                if (isLXTFormat) {
                    // Store LXT data for rendering
                    this.currentLXTData = report;
                    this.currentLXTSampleIndex = 0;
                    this.totalLXTSamples = report.length;
                    this.currentOutputTokenIndex = 0;
                    this.lockedLXTOutputToken = null;
                    
                    // Create container for LXT visualization (will be rendered after DOM insertion)
                    reportContent = '<div id="lxt-visualization-container"></div>';
                    needsExpansion = false; // LXT doesn't need expansion as it's interactive
                    
                    // Store for potential future use
                    this.currentDiagnosticReports[index] = {
                        full: '',
                        preview: '',
                        type: 'lxt',
                        data: report
                    };
                } else if (isSHAPFormat) {
                    // Handle SHAP visualization with two-column layout
                    // Store the data for rendering after HTML is inserted
                    this.currentSHAPData = report; // report is already the array of SHAP samples
                    this.totalSHAPSamples = this.currentSHAPData.length;
                    this.currentSHAPSampleIndex = 0;
                    this.lockedInputToken = null;
                    
                    reportContent = '<div id="shap-visualization-container"></div>';
                    needsExpansion = false;
                    
                    this.currentDiagnosticReports[index] = {
                        full: '',
                        preview: '',
                        type: 'shap',
                        data: report
                    };
                } else if (isSHAPEncoderFormat) {
                    // Handle SHAP Encoder visualization for cross-encoder re-ranking
                    this.currentSHAPEncoderData = report;
                    this.totalSHAPEncoderSamples = report.length;
                    this.currentSHAPEncoderSampleIndex = 0;
                    this.currentSHAPEncoderDocIndex = 0;
                    this.totalSHAPEncoderDocs = report.length > 0 ? report[0].attributions.length : 0;
                    
                    reportContent = '<div id="shap-encoder-visualization-container"></div>';
                    needsExpansion = false;
                    
                    this.currentDiagnosticReports[index] = {
                        full: '',
                        preview: '',
                        type: 'shap_encoder',
                        data: report
                    };
                } else if (isIGEncoderFormat) {
                    // Handle IG Encoder visualization for cross-encoder re-ranking
                    this.currentIGEncoderData = report;
                    this.totalIGEncoderSamples = report.length;
                    this.currentIGEncoderSampleIndex = 0;
                    this.currentIGEncoderDocIndex = 0;
                    this.totalIGEncoderDocs = report.length > 0 ? report[0].attributions.length : 0;
                    
                    reportContent = '<div id="ig-encoder-visualization-container"></div>';
                    needsExpansion = false;
                    
                    this.currentDiagnosticReports[index] = {
                        full: '',
                        preview: '',
                        type: 'integrated_gradients_encoder',
                        data: report
                    };
                } else {
                    // Handle text format (existing logic)
                    const reportText = typeof report === 'string' ? report : JSON.stringify(report, null, 2);
                    this.currentDiagnosticReports[index] = {
                        full: reportText,
                        preview: reportText.length > 500 ? reportText.substring(0, 500) + '...' : reportText,
                        type: 'text'
                    };
                    
                    reportContent = this.currentDiagnosticReports[index].preview;
                    needsExpansion = reportText.length > 500;
                }
                
                // Format method name for display
                const displayName = methodName.replace(/_/g, ' ')
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
                
                diagnosticsHtml += `
                    <div class="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden mb-6">
                        <div class="p-4 border-b border-gray-200 bg-gray-50">
                            <div class="flex items-center justify-between">
                                <h4 class="font-medium text-gray-800 flex items-center">
                                    <i class="fas fa-stethoscope mr-2 text-purple-500"></i>
                                    ${displayName}
                                    ${isLXTFormat ? `
                                        <span class="ml-2 px-2 py-1 text-xs font-medium bg-purple-100 text-purple-700 rounded-full">
                                            Token Relevance
                                        </span>
                                    ` : ''}
                                    ${isSHAPFormat ? `
                                        <span class="ml-2 px-2 py-1 text-xs font-medium bg-blue-100 text-blue-700 rounded-full">
                                            Attribution Analysis
                                        </span>
                                    ` : ''}
                                    ${isSHAPEncoderFormat ? `
                                        <span class="ml-2 px-2 py-1 text-xs font-medium bg-teal-100 text-teal-700 rounded-full">
                                            Re-Ranker Attribution
                                        </span>
                                    ` : ''}
                                    ${isIGEncoderFormat ? `
                                        <span class="ml-2 px-2 py-1 text-xs font-medium bg-purple-100 text-purple-700 rounded-full">
                                            Integrated Gradients
                                        </span>
                                    ` : ''}
                                </h4>
                                ${isLXTFormat ? `
                                    <div class="flex items-center space-x-3 text-xs text-gray-600">
                                        <span class="font-medium">Legend:</span>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(103,169,207); border: 1px solid #4b5563;"></span>
                                            <span>Positive</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(247,247,247); border: 1px solid #4b5563;"></span>
                                            <span>Neutral</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(214,96,77); border: 1px solid #4b5563;"></span>
                                            <span>Negative</span>
                                        </div>
                                    </div>
                                ` : ''}
                                ${isSHAPFormat ? `
                                    <div class="flex items-center space-x-3 text-xs text-gray-600">
                                        <span class="font-medium">Legend:</span>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(103,169,207); border: 1px solid #4b5563;"></span>
                                            <span>Positive</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(247,247,247); border: 1px solid #4b5563;"></span>
                                            <span>Neutral</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(214,96,77); border: 1px solid #4b5563;"></span>
                                            <span>Negative</span>
                                        </div>
                                    </div>
                                ` : ''}
                                ${isSHAPEncoderFormat ? `
                                    <div class="flex items-center space-x-3 text-xs text-gray-600">
                                        <span class="font-medium">Legend:</span>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(103,169,207); border: 1px solid #4b5563;"></span>
                                            <span>Positive</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(247,247,247); border: 1px solid #4b5563;"></span>
                                            <span>Neutral</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(214,96,77); border: 1px solid #4b5563;"></span>
                                            <span>Negative</span>
                                        </div>
                                    </div>
                                ` : ''}
                                ${isIGEncoderFormat ? `
                                    <div class="flex items-center space-x-3 text-xs text-gray-600">
                                        <span class="font-medium">Legend:</span>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(103,169,207); border: 1px solid #4b5563;"></span>
                                            <span>Positive</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(247,247,247); border: 1px solid #4b5563;"></span>
                                            <span>Neutral</span>
                                        </div>
                                        <div class="flex items-center space-x-1">
                                            <span class="inline-block w-3 h-3 rounded" style="background-color: rgb(214,96,77); border: 1px solid #4b5563;"></span>
                                            <span>Negative</span>
                                        </div>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                        <div class="p-4">
                            <div class="bg-gray-50 rounded-lg p-4">
                                <div id="diagnosis-report-${index}" class="${isLXTFormat || isSHAPFormat || isSHAPEncoderFormat || isIGEncoderFormat ? 'text-sm leading-relaxed' : 'text-sm text-gray-700 whitespace-pre-wrap leading-relaxed font-sans'}" data-is-lxt="${isLXTFormat}" data-is-shap="${isSHAPFormat}" data-is-shap-encoder="${isSHAPEncoderFormat}" data-is-ig-encoder="${isIGEncoderFormat}" data-report-index="${index}">${reportContent}</div>
                                ${needsExpansion ? `
                                    <button id="diagnosis-toggle-${index}" 
                                            onclick="toggleDiagnosisExpansion(${index})" 
                                            class="mt-3 text-purple-600 hover:text-purple-800 text-sm font-medium">
                                        <i class="fas fa-chevron-down mr-1"></i>Show Full Report
                                    </button>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                `;
            });
            
            diagnosisContainer.innerHTML = diagnosticsHtml;
            
            // After rendering, automatically render LXT, SHAP, SHAP Encoder, and IG Encoder visualizations
            // Use a small delay to ensure DOM is fully updated
            setTimeout(() => {
                if (this.currentLXTData && this.currentLXTData.length > 0) {
                    this.renderLXTVisualization();
                }
                if (this.currentSHAPData && this.currentSHAPData.length > 0) {
                    this.renderSHAPVisualization();
                }
                if (this.currentSHAPEncoderData && this.currentSHAPEncoderData.length > 0) {
                    this.renderSHAPEncoderVisualization();
                }
                if (this.currentIGEncoderData && this.currentIGEncoderData.length > 0) {
                    this.renderIGEncoderVisualization();
                }
            }, 100);
        } else {
            // Show appropriate message based on job status
            let message, icon, title;
            if (job.status === 'FAILED') {
                icon = 'fa-exclamation-triangle';
                title = 'Job Failed';
                message = 'Diagnostics are not available because the job failed.';
            } else if (job.status === 'RUNNING' || job.status === 'QUEUED') {
                icon = 'fa-clock';
                title = 'Job In Progress';
                message = 'Diagnostics will be available after the job completes.';
            } else {
                icon = 'fa-info-circle';
                title = 'No Diagnostics Available';
                message = 'No diagnostic results are available for this job.';
            }
            
            diagnosisContainer.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas ${icon} text-4xl text-gray-300 mb-4"></i>
                    <h3 class="text-lg font-medium text-gray-600 mb-2">${title}</h3>
                    <p class="text-gray-500">${message}</p>
                </div>
            `;
        }
    }

    renderLXTVisualization() {
        const container = document.getElementById('lxt-visualization-container');
        if (!container) return;
        
        if (!this.currentLXTData || this.currentLXTData.length === 0) {
            container.innerHTML = '<p class="text-gray-500">No LXT data available</p>';
            return;
        }
        
        let html = '<div class="lxt-visualization">';
        
        // Add sample navigation if multiple samples
        if (this.totalLXTSamples > 1) {
            html += `
                <div class="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div class="flex items-center space-x-3">
                        <h4 class="font-medium text-gray-800">Sample Navigation</h4>
                        <span id="lxt-sample-counter" class="text-sm text-gray-600">1 of ${this.totalLXTSamples}</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <button id="lxt-prev-sample-btn" 
                                onclick="contentManager.navigateLXTSample(-1)"
                                class="px-3 py-1.5 bg-white border border-gray-300 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed">
                            <i class="fas fa-chevron-left mr-1"></i>Previous
                        </button>
                        <button id="lxt-next-sample-btn" 
                                onclick="contentManager.navigateLXTSample(1)"
                                class="px-3 py-1.5 bg-white border border-gray-300 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed">
                            Next<i class="fas fa-chevron-right ml-1"></i>
                        </button>
                    </div>
                </div>
            `;
        }
        
        // Container for current sample
        html += '<div id="lxt-current-sample"></div>';
        html += '</div>';
        
        container.innerHTML = html;
        
        // Render the first sample
        this.renderCurrentLXTSample();
    }
    
    navigateLXTSample(direction) {
        const newIndex = this.currentLXTSampleIndex + direction;
        if (newIndex < 0 || newIndex >= this.totalLXTSamples) return;
        this.currentLXTSampleIndex = newIndex;
        this.lockedLXTOutputToken = null;   // Clear locked token when navigating
        this.updateLXTSampleDisplay();      // Update navigation UI
        this.renderCurrentLXTSample();      // Render the new sample
    }
    
    renderCurrentLXTSample() {
        const container = document.getElementById('lxt-current-sample');
        if (!container || !this.currentLXTData) return;
        
        const sampleData = this.currentLXTData[this.currentLXTSampleIndex];
        const attributionMatrix = sampleData.attribution_matrix || [];
        
        // Get output tokens from the array (each element has generated_token and token_relevance)
        // Filter out any undefined/null values
        const outputTokens = attributionMatrix
            .map(item => item.generated_token || item.output_token)
            .filter(token => token !== undefined && token !== null);
        
        // Get input tokens from the first output token's relevance (all should have the same input tokens)
        const inputTokens = attributionMatrix.length > 0 && attributionMatrix[0].token_relevance
            ? attributionMatrix[0].token_relevance.map(([token, score]) => token).filter(token => token !== undefined && token !== null)
            : [];
        
        // Calculate average importance for each output token (for optional coloring)
        const avgImportances = {};
        
        attributionMatrix.forEach(item => {
            const outputToken = item.generated_token || item.output_token;
            if (!outputToken || !item.token_relevance) return; // Skip invalid items
            const scores = item.token_relevance.map(([token, score]) => score);
            const avg = scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
            avgImportances[outputToken] = avg;
        });
        
        // Use magnitude-based color mapping
        const allAvgScores = Object.values(avgImportances);
        const avgColorMap = this.generateMatplotlibColorMap(allAvgScores, 'RdBu_r');
        
        // Build vertical layout (stacked containers)
        let html = `
            <div class="space-y-6">
                <!-- Input Document -->
                <div class="bg-gray-50 rounded-lg border border-gray-200 p-5" style="overflow: visible;">
                    <h4 class="font-medium text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-file-text mr-2 text-blue-500"></i>
                        Input
                        <span class="ml-2 text-xs text-gray-500">(Highlighted by output token relevance)</span>
                    </h4>
                    <div class="bg-white rounded p-4 border border-gray-200" style="max-height: 600px; overflow-y: auto; overflow-x: visible;">
                        <div class="flex flex-wrap gap-1" id="lxt-input-tokens" style="overflow: visible;">
        `;
        
        // Render input tokens (will be colored on hover)
        inputTokens.forEach((token, index) => {
            if (!token) return; // Skip null/undefined tokens
            const cleanToken = token.replace(/\u2581/g, '').replace(/_/g, '').replace(/▁/g, '');
            
            html += `
                <span class="lxt-input-token px-2 py-1 rounded transition-all duration-200"
                    data-token-index="${index}"
                    data-token="${this.escapeHtml(token)}"
                    data-current-score="0.0000"
                    onmouseover="contentManager.showLXTTooltip(this, event)"
                    onmouseout="contentManager.hideLXTTooltip()"
                    style="background-color: #f3f4f6; color: #374151;">
                    ${this.escapeHtml(cleanToken)}
                </span>
            `;
        });
        
        html += `
                        </div>
                    </div>
                </div>
                
                <!-- Generated Output -->
                <div class="bg-gray-50 rounded-lg border border-gray-200 p-5" style="overflow: visible;">
                    <h4 class="font-medium text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-file-alt mr-2 text-green-500"></i>
                        Output
                        <span class="ml-2 text-xs text-gray-500">(Hover/click tokens to see which inputs influenced it)</span>
                    </h4>
                    <div class="bg-white rounded p-4 border border-gray-200" style="max-height: 600px; overflow-y: auto; position: relative;">
                        <div class="flex flex-wrap gap-1" id="lxt-output-tokens" style="position: relative;">
        `;
        
        // Render output tokens with magnitude-based coloring and make them hoverable
        outputTokens.forEach((token, index) => {
            if (!token) return; // Skip null/undefined tokens
            const cleanToken = token.replace(/\u2581/g, '').replace(/_/g, '').replace(/▁/g, '');
            const avgScore = avgImportances[token];
            const color = avgColorMap[avgScore] || '#f3f4f6';
            const textColor = this.getContrastTextColor(color);
            
            html += `
                <span class="lxt-output-token px-2 py-1 rounded cursor-pointer transition-all duration-200"
                      data-token-index="${index}"
                      data-token="${this.escapeHtml(token)}"
                      onmouseover="contentManager.hoverLXTOutputToken(${index}, false)"
                      onmouseout="contentManager.clearLXTHoverHighlight()"
                      onclick="contentManager.clickLXTOutputToken(${index})"
                      style="background-color: ${color}; color: ${textColor};">
                    ${this.escapeHtml(cleanToken)}
                </span>
            `;
        });
        
        html += `
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tooltip for LXT scores -->
            <div id="lxt-global-tooltip" style="position: fixed; display: none; background-color: rgba(0, 0, 0, 0.8); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 1000;"></div>
        `;
        
        container.innerHTML = html;
        
        // Store tokens for interaction
        this.currentLXTInputTokens = inputTokens;
        this.currentLXTOutputTokens = outputTokens;
    }
    
    showLXTTooltip(element, event) {
        const score = element.dataset.currentScore || '0.0000';
        
        // Don't show tooltip if no output token is selected (default state)
        if (score === '0.0000' && this.lockedLXTOutputToken === null) return;
        
        const tooltip = document.getElementById('lxt-global-tooltip');
        if (!tooltip) return;
        
        tooltip.textContent = `Score: ${score}`;
        tooltip.style.display = 'block';
        
        // Position tooltip above the element
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        // Center horizontally above the token
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltipRect.width / 2}px`;
        tooltip.style.top = `${rect.top - tooltipRect.height - 8}px`;
    }
    
    hideLXTTooltip() {
        const tooltip = document.getElementById('lxt-global-tooltip');
        if (tooltip) tooltip.style.display = 'none';
    }
    
    hoverLXTOutputToken(tokenIndex, isLocked) {
        // If there's a locked token and this is just a hover, ignore
        if (this.lockedLXTOutputToken !== null && !isLocked) return;
        
        const outputTokenElements = document.querySelectorAll('.lxt-output-token');
        const outputToken = outputTokenElements[tokenIndex].dataset.token;
        
        if (!this.currentLXTData || !this.currentLXTInputTokens) return;
        
        const sampleData = this.currentLXTData[this.currentLXTSampleIndex];
        const attributionMatrix = sampleData.attribution_matrix || [];
        
        // Get the relevance data for this specific output token index
        const attributionItem = attributionMatrix[tokenIndex];
        if (!attributionItem) return;
        
        // Extract scores from the token_relevance array of tuples
        const tokenRelevance = attributionItem.token_relevance || [];
        const allScores = tokenRelevance.map(([token, score]) => score);
        
        // Use magnitude-based color mapping
        const colorMap = this.generateMatplotlibColorMap(allScores, 'RdBu_r');
        
        // Highlight input tokens based on their relevance scores for this output
        const inputTokenElements = document.querySelectorAll('.lxt-input-token');
        inputTokenElements.forEach((element, index) => {
            const [inputToken, score] = tokenRelevance[index] || ['', 0];
            const color = colorMap[score] || '#f3f4f6';
            const textColor = this.getContrastTextColor(color);
            
            element.style.backgroundColor = color;
            element.style.color = textColor;
            
            // Store score as data attribute for tooltip access
            element.dataset.currentScore = score.toFixed(4);
        });
        
        // Highlight the hovered/locked output token
        outputTokenElements.forEach((el, idx) => {
            if (idx === tokenIndex) {
                el.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.5)';
                el.style.transform = 'scale(1.05)';
            } else if (!isLocked) {
                el.style.boxShadow = '';
                el.style.transform = '';
            }
        });
    }
    
    clearLXTHoverHighlight() {
        // If there's a locked token, don't clear
        if (this.lockedLXTOutputToken !== null) return;
        
        // Reset all input tokens to default
        const inputTokenElements = document.querySelectorAll('.lxt-input-token');
        inputTokenElements.forEach(element => {
            element.style.backgroundColor = '#f3f4f6';
            element.style.color = '#374151';
        });
        
        // Reset output token highlighting
        const outputTokenElements = document.querySelectorAll('.lxt-output-token');
        outputTokenElements.forEach(el => {
            el.style.boxShadow = '';
            el.style.transform = '';
        });
    }
    
    clickLXTOutputToken(tokenIndex) {
        if (this.lockedLXTOutputToken === tokenIndex) {
            // Clicking the same token unlocks it
            this.lockedLXTOutputToken = null;
            this.clearLXTHoverHighlight();
            
            // Also clear output token highlighting
            const outputTokenElements = document.querySelectorAll('.lxt-output-token');
            outputTokenElements.forEach(el => {
                el.style.boxShadow = '';
                el.style.transform = '';
            });
        } else {
            // Clicking a different token locks it
            // First clear all output token highlighting
            const outputTokenElements = document.querySelectorAll('.lxt-output-token');
            outputTokenElements.forEach(el => {
                el.style.boxShadow = '';
                el.style.transform = '';
            });
            
            // Lock to this token
            this.lockedLXTOutputToken = tokenIndex;
            this.hoverLXTOutputToken(tokenIndex, true);
        }
    }
    
    updateLXTSampleDisplay() {
        // Update sample counter
        const counterElement = document.getElementById('lxt-sample-counter');
        if (counterElement) {
            counterElement.textContent = `${this.currentLXTSampleIndex + 1} of ${this.totalLXTSamples}`;
        }
        
        // Update button states
        const prevBtn = document.getElementById('lxt-prev-sample-btn');
        const nextBtn = document.getElementById('lxt-next-sample-btn');
        
        if (prevBtn) {
            prevBtn.disabled = this.currentLXTSampleIndex === 0;
            if (prevBtn.disabled) {
                prevBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                prevBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
        
        if (nextBtn) {
            nextBtn.disabled = this.currentLXTSampleIndex >= this.totalLXTSamples - 1;
            if (nextBtn.disabled) {
                nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
    }
    
    renderSHAPVisualization() {
        const container = document.getElementById('shap-visualization-container');
        if (!container) {
            console.error('SHAP container element not found in DOM');
            return;
        }
        
        if (!this.currentSHAPData || this.currentSHAPData.length === 0) {
            console.warn('No SHAP data available');
            container.innerHTML = '<span class="text-gray-500 italic">No SHAP data available</span>';
            return;
        }
        
        let html = '<div class="shap-visualization">';
        
        // Add sample navigation if multiple samples
        if (this.totalSHAPSamples > 1) {
            html += `
                <div class="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div class="flex items-center space-x-3">
                        <h4 class="font-medium text-gray-800">Input Sample Navigation</h4>
                        <span id="shap-sample-counter" class="text-sm text-gray-600">1 of ${this.totalSHAPSamples}</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <button id="shap-prev-sample-btn" 
                                onclick="contentManager.navigateSHAPSample(-1)"
                                class="px-3 py-1 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                                disabled>
                            <i class="fas fa-chevron-left"></i> Previous
                        </button>
                        <button id="shap-next-sample-btn" 
                                onclick="contentManager.navigateSHAPSample(1)"
                                class="px-3 py-1 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed">
                            Next <i class="fas fa-chevron-right"></i>
                        </button>
                    </div>
                </div>
            `;
        }
        
        // Container for current sample
        html += '<div id="shap-current-sample"></div>';
        html += '</div>';
        
        container.innerHTML = html;
        
        // Render the first sample
        this.renderCurrentSHAPSample();
    }
    
    navigateSHAPSample(direction) {
        const newIndex = this.currentSHAPSampleIndex + direction;
        if (newIndex < 0 || newIndex >= this.totalSHAPSamples) {
            return;
        }
        this.currentSHAPSampleIndex = newIndex;
        this.lockedOutputToken = null;   // Clear locked token when navigating
        this.updateSHAPSampleDisplay(); // Update navigation UI
        this.renderCurrentSHAPSample(); // Render the new sample
    }
    
    renderCurrentSHAPSample() {
        const container = document.getElementById('shap-current-sample');
        if (!container || !this.currentSHAPData) return;
        
        const sampleData = this.currentSHAPData[this.currentSHAPSampleIndex];
        const attributionMatrix = sampleData.attribution_matrix || [];
        
        // Get output tokens from the array (each element has output_token and input_attributions)
        const outputTokens = attributionMatrix.map(item => item.output_token || item.generated_token);
        // Get input tokens from the first output token's attributions (all should have the same input tokens)
        const inputTokens = attributionMatrix.length > 0 
            ? attributionMatrix[0].input_attributions.map(([token, score]) => token)
            : [];
        
        // Calculate average importance for each output token (for optional coloring)
        const avgImportances = {};
        
        attributionMatrix.forEach(item => {
            const outputToken = item.output_token || item.generated_token;
            const scores = item.input_attributions.map(([token, score]) => score);
            // Calculate actual mean (preserving sign) to show positive vs negative attribution
            const avg = scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
            avgImportances[outputToken] = avg;
        });
        
        // Use magnitude-based color mapping (like LXT) for more nuanced coloring
        const allAvgScores = Object.values(avgImportances);
        const avgColorMap = this.generateMatplotlibColorMap(allAvgScores, 'RdBu_r');
        
        // Build vertical layout (stacked containers)
        let html = `
            <div class="space-y-6">
                <!-- Input Document -->
                <div class="bg-gray-50 rounded-lg border border-gray-200 p-5" style="overflow: visible;">
                    <h4 class="font-medium text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-file-text mr-2 text-blue-500"></i>
                        Input
                        <span class="ml-2 text-xs text-gray-500">(Highlighted by output token attribution)</span>
                    </h4>
                    <div class="bg-white rounded p-4 border border-gray-200" style="max-height: 600px; overflow-y: auto; overflow-x: visible;">
                        <div class="flex flex-wrap gap-1" id="shap-input-tokens" style="overflow: visible;">
        `;
        
        // Render input tokens (will be colored on hover)
        inputTokens.forEach((token, index) => {
            const cleanToken = token.replace(/\u2581/g, '').replace(/_/g, '');
            
            html += `
                <span class="shap-input-token px-2 py-1 rounded transition-all duration-200"
                    data-token-index="${index}"
                    data-token="${this.escapeHtml(token)}"
                    data-current-score="0.0000"
                    onmouseover="contentManager.showSHAPTooltip(this, event)"
                    onmouseout="contentManager.hideSHAPTooltip()"
                    style="background-color: #f3f4f6; color: #374151;">
                    ${this.escapeHtml(cleanToken)}
                </span>
            `;
        });
        
        html += `
                        </div>
                    </div>
                </div>
                
                <!-- Generated Summary -->
                <div class="bg-gray-50 rounded-lg border border-gray-200 p-5" style="overflow: visible;">
                    <h4 class="font-medium text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-file-alt mr-2 text-green-500"></i>
                        Output
                        <span class="ml-2 text-xs text-gray-500">(Hover/click tokens to see which inputs influenced it)</span>
                    </h4>
                    <div class="bg-white rounded p-4 border border-gray-200" style="max-height: 600px; overflow-y: auto; position: relative;">
                        <div class="flex flex-wrap gap-1" id="shap-output-tokens" style="position: relative;">
        `;
        
        // Render output tokens with magnitude-based coloring and make them hoverable
        outputTokens.forEach((token, index) => {
            const cleanToken = token.replace(/\u2581/g, '').replace(/_/g, '');
            const avgScore = avgImportances[token];
            const color = avgColorMap[avgScore] || '#f3f4f6';
            const textColor = this.getContrastTextColor(color);
            
            html += `
                <span class="shap-output-token px-2 py-1 rounded cursor-pointer transition-all duration-200"
                      data-token-index="${index}"
                      data-token="${this.escapeHtml(token)}"
                      onmouseover="contentManager.hoverOutputToken(${index}, false)"
                      onmouseout="contentManager.clearHoverHighlight()"
                      onclick="contentManager.clickOutputToken(${index})"
                      style="background-color: ${color}; color: ${textColor};">
                    ${this.escapeHtml(cleanToken)}
                </span>
            `;
        });
        
        html += `
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tooltip for SHAP scores -->
            <div id="shap-global-tooltip" style="position: fixed; display: none; background-color: rgba(0, 0, 0, 0.8); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 1000;"></div>
        `;
        
        container.innerHTML = html;
        
        // Store tokens for interaction
        this.currentSHAPInputTokens = inputTokens;
        this.currentSHAPOutputTokens = outputTokens;
    }
    
    showSHAPTooltip(element, event) {
        // Only show tooltip if an output token is actively selected (hovered or locked)
        const score = element.dataset.currentScore || '0.0000';
        
        // Don't show tooltip if no output token is selected (default state)
        if (score === '0.0000' && this.lockedOutputToken === null) {
            return;
        }
        
        const tooltip = document.getElementById('shap-global-tooltip');
        if (!tooltip) return;
        
        tooltip.textContent = `Score: ${score}`;
        tooltip.style.display = 'block';
        
        // Position tooltip above the element
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        // Center horizontally above the token
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltipRect.width / 2}px`;
        tooltip.style.top = `${rect.top - tooltipRect.height - 8}px`;
    }
    
    hideSHAPTooltip() {
        const tooltip = document.getElementById('shap-global-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    hoverOutputToken(tokenIndex, isLocked) {
        // If there's a locked token and this is just a hover, ignore
        if (this.lockedOutputToken !== null && !isLocked) return;
        
        const outputTokenElements = document.querySelectorAll('.shap-output-token');
        const outputToken = outputTokenElements[tokenIndex].dataset.token;
        
        if (!this.currentSHAPData || !this.currentSHAPInputTokens) return;
        
        const sampleData = this.currentSHAPData[this.currentSHAPSampleIndex];
        const attributionMatrix = sampleData.attribution_matrix || [];
        
        // Get the attribution data for this specific output token index
        const attributionItem = attributionMatrix[tokenIndex];
        if (!attributionItem) return;
        
        // Extract scores from the input_attributions array of tuples
        const inputAttributions = attributionItem.input_attributions || [];
        const allScores = inputAttributions.map(([token, score]) => score);
        
        // Use magnitude-based color mapping for input tokens (like LXT)
        const colorMap = this.generateMatplotlibColorMap(allScores, 'RdBu_r');
        
        // Highlight input tokens based on their attribution scores for this output
        const inputTokenElements = document.querySelectorAll('.shap-input-token');
        inputTokenElements.forEach((element, index) => {
            const [inputToken, score] = inputAttributions[index] || ['', 0];
            const color = colorMap[score] || '#f3f4f6';
            const textColor = this.getContrastTextColor(color);
            
            element.style.backgroundColor = color;
            element.style.color = textColor;
            
            // Store score as data attribute for tooltip access
            element.dataset.currentScore = score.toFixed(4);
        });
        
        // Highlight the hovered/locked output token
        outputTokenElements.forEach((el, idx) => {
            if (idx === tokenIndex) {
                el.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.5)';
                el.style.transform = 'scale(1.05)';
            } else if (!isLocked) {
                el.style.boxShadow = '';
                el.style.transform = '';
            }
        });
    }
    
    clearHoverHighlight() {
        // If there's a locked token, don't clear
        if (this.lockedOutputToken !== null) return;
        
        // Reset all input tokens to default
        const inputTokenElements = document.querySelectorAll('.shap-input-token');
        inputTokenElements.forEach(element => {
            element.style.backgroundColor = '#f3f4f6';
            element.style.color = '#374151';
        });
        
        // Reset output token highlighting
        const outputTokenElements = document.querySelectorAll('.shap-output-token');
        outputTokenElements.forEach(el => {
            el.style.boxShadow = '';
            el.style.transform = '';
        });
    }
    
    clickOutputToken(tokenIndex) {
        if (this.lockedOutputToken === tokenIndex) {
            // Clicking the same token unlocks it
            this.lockedOutputToken = null;
            this.clearHoverHighlight();
            
            // Also clear output token highlighting
            const outputTokenElements = document.querySelectorAll('.shap-output-token');
            outputTokenElements.forEach(el => {
                el.style.boxShadow = '';
                el.style.transform = '';
            });
        } else {
            // Clicking a different token locks it
            // First clear all output token highlighting
            const outputTokenElements = document.querySelectorAll('.shap-output-token');
            outputTokenElements.forEach(el => {
                el.style.boxShadow = '';
                el.style.transform = '';
            });
            
            // Lock to this token
            this.lockedOutputToken = tokenIndex;
            this.hoverOutputToken(tokenIndex, true);
        }
    }
    
    updateSHAPSampleDisplay() {
        // Update sample counter
        const counterElement = document.getElementById('shap-sample-counter');
        if (counterElement) {
            counterElement.textContent = `${this.currentSHAPSampleIndex + 1} of ${this.totalSHAPSamples}`;
        }
        
        // Update button states
        const prevBtn = document.getElementById('shap-prev-sample-btn');
        const nextBtn = document.getElementById('shap-next-sample-btn');
        
        if (prevBtn) {
            prevBtn.disabled = this.currentSHAPSampleIndex === 0;
            if (prevBtn.disabled) {
                prevBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                prevBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
        
        if (nextBtn) {
            nextBtn.disabled = this.currentSHAPSampleIndex >= this.totalSHAPSamples - 1;
            if (nextBtn.disabled) {
                nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
    }

    // ==================== SHAP Encoder Visualization Methods ====================
    
    renderSHAPEncoderVisualization() {
        const container = document.getElementById('shap-encoder-visualization-container');
        if (!container || !this.currentSHAPEncoderData || this.currentSHAPEncoderData.length === 0) {
            return;
        }
        
        // Reset to first sample and document
        this.currentSHAPEncoderSampleIndex = 0;
        const firstSample = this.currentSHAPEncoderData[0];
        this.totalSHAPEncoderDocs = firstSample.attributions ? firstSample.attributions.length : 0;
        this.currentSHAPEncoderDocIndex = 0;
        
        // Create HTML structure
        container.innerHTML = `
            <!-- Sample Navigation (Query Level) -->
            <div class="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div class="flex items-center space-x-2">
                    <span class="text-sm text-gray-600 font-medium">Query:</span>
                    <span id="shap-encoder-sample-counter" class="text-sm text-gray-600">1 of ${this.totalSHAPEncoderSamples}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <button 
                        id="shap-encoder-prev-sample-btn"
                        onclick="window.contentManager.navigateSHAPEncoderSample(-1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <i class="fas fa-chevron-left mr-1"></i>Previous Query
                    </button>
                    <button 
                        id="shap-encoder-next-sample-btn"
                        onclick="window.contentManager.navigateSHAPEncoderSample(1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Next Query<i class="fas fa-chevron-right ml-1"></i>
                    </button>
                </div>
            </div>
            
            <!-- Document Navigation -->
            <div class="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div class="flex items-center space-x-2">
                    <span class="text-sm text-gray-600 font-medium">Document:</span>
                    <span id="shap-encoder-doc-counter" class="text-sm text-gray-600">1 of ${this.totalSHAPEncoderDocs}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <button 
                        id="shap-encoder-prev-doc-btn"
                        onclick="window.contentManager.navigateSHAPEncoderDocument(-1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <i class="fas fa-chevron-left mr-1"></i>Previous Document
                    </button>
                    <button 
                        id="shap-encoder-next-doc-btn"
                        onclick="window.contentManager.navigateSHAPEncoderDocument(1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Next Document<i class="fas fa-chevron-right ml-1"></i>
                    </button>
                </div>
            </div>
            
            <!-- Visualization Area -->
            <div id="shap-encoder-content-area"></div>
        `;
        
        // Render the first view
        this.renderCurrentSHAPEncoderView();
    }
    
    navigateSHAPEncoderSample(direction) {
        const newIndex = this.currentSHAPEncoderSampleIndex + direction;
        if (newIndex >= 0 && newIndex < this.totalSHAPEncoderSamples) {
            this.currentSHAPEncoderSampleIndex = newIndex;
            
            // Reset document index and update total docs for new sample
            const currentSample = this.currentSHAPEncoderData[this.currentSHAPEncoderSampleIndex];
            this.totalSHAPEncoderDocs = currentSample.attributions ? currentSample.attributions.length : 0;
            this.currentSHAPEncoderDocIndex = 0;
            
            this.renderCurrentSHAPEncoderView();
        }
    }
    
    navigateSHAPEncoderDocument(direction) {
        const newIndex = this.currentSHAPEncoderDocIndex + direction;
        if (newIndex >= 0 && newIndex < this.totalSHAPEncoderDocs) {
            this.currentSHAPEncoderDocIndex = newIndex;
            this.renderCurrentSHAPEncoderView();
        }
    }
    
    renderCurrentSHAPEncoderView() {
        const contentArea = document.getElementById('shap-encoder-content-area');
        if (!contentArea) return;
        
        // Update navigation counters
        const sampleCounter = document.getElementById('shap-encoder-sample-counter');
        const docCounter = document.getElementById('shap-encoder-doc-counter');
        
        if (sampleCounter) {
            sampleCounter.textContent = `${this.currentSHAPEncoderSampleIndex + 1} of ${this.totalSHAPEncoderSamples}`;
        }
        
        if (docCounter) {
            docCounter.textContent = `${this.currentSHAPEncoderDocIndex + 1} of ${this.totalSHAPEncoderDocs}`;
        }
        
        // Update button states
        const prevSampleBtn = document.getElementById('shap-encoder-prev-sample-btn');
        const nextSampleBtn = document.getElementById('shap-encoder-next-sample-btn');
        const prevDocBtn = document.getElementById('shap-encoder-prev-doc-btn');
        const nextDocBtn = document.getElementById('shap-encoder-next-doc-btn');
        
        if (prevSampleBtn) {
            prevSampleBtn.disabled = this.currentSHAPEncoderSampleIndex === 0;
            prevSampleBtn.classList.toggle('opacity-50', prevSampleBtn.disabled);
            prevSampleBtn.classList.toggle('cursor-not-allowed', prevSampleBtn.disabled);
        }
        
        if (nextSampleBtn) {
            nextSampleBtn.disabled = this.currentSHAPEncoderSampleIndex >= this.totalSHAPEncoderSamples - 1;
            nextSampleBtn.classList.toggle('opacity-50', nextSampleBtn.disabled);
            nextSampleBtn.classList.toggle('cursor-not-allowed', nextSampleBtn.disabled);
        }
        
        if (prevDocBtn) {
            prevDocBtn.disabled = this.currentSHAPEncoderDocIndex === 0;
            prevDocBtn.classList.toggle('opacity-50', prevDocBtn.disabled);
            prevDocBtn.classList.toggle('cursor-not-allowed', prevDocBtn.disabled);
        }
        
        if (nextDocBtn) {
            nextDocBtn.disabled = this.currentSHAPEncoderDocIndex >= this.totalSHAPEncoderDocs - 1;
            nextDocBtn.classList.toggle('opacity-50', nextDocBtn.disabled);
            nextDocBtn.classList.toggle('cursor-not-allowed', nextDocBtn.disabled);
        }
        
        // Get current data
        const currentSample = this.currentSHAPEncoderData[this.currentSHAPEncoderSampleIndex];
        if (!currentSample || !currentSample.attributions || currentSample.attributions.length === 0) {
            contentArea.innerHTML = '<div class="text-gray-500 text-center py-8">No attribution data available</div>';
            return;
        }
        
        const currentDoc = currentSample.attributions[this.currentSHAPEncoderDocIndex];
        const queryAttrs = currentDoc.query_attributions || [];
        const docAttrs = currentDoc.document_attributions || [];
        
        // Collect all scores for colormap generation
        const allScores = [
            ...queryAttrs.map(attr => attr[1]),
            ...docAttrs.map(attr => attr[1])
        ];
        const colorMap = this.generateMatplotlibColorMap(allScores, 'RdBu_r');
        
        // Render context panel
        const contextHtml = `
            <div class="bg-purple-50 rounded-lg p-4 mb-4 border border-gray-300">
                <div class="grid grid-cols-3 gap-4 text-sm">
                    <div>
                        <span class="font-medium text-gray-700">Query ID:</span>
                        <span class="ml-2 text-gray-900 font-mono">${currentSample.query_id || 'N/A'}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Document ID:</span>
                        <span class="ml-2 text-gray-900 font-mono">${currentDoc.document_id || 'N/A'}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Rerank Score:</span>
                        <span class="ml-2 text-gray-900 font-bold">${currentDoc.rerank_score?.toFixed(4) || 'N/A'}</span>
                    </div>
                </div>
                <div class="mt-3 pt-3 border-t border-gray-200">
                    <div class="font-medium text-gray-700 mb-1">Query Text:</div>
                    <div class="text-gray-900 bg-white rounded p-2 text-sm border border-gray-300">${this.escapeHtml(currentSample.query)}</div>
                </div>
                <div class="mt-2">
                    <button 
                        onclick="showDocumentModal('${this.escapeHtml(currentDoc.document_text).replace(/'/g, '&apos;')}', '${currentDoc.document_id || 'N/A'}')"
                        class="text-purple-600 hover:text-purple-800 text-sm font-medium"
                    >
                        <i class="fas fa-file-alt mr-1"></i>View Full Document
                    </button>
                </div>
            </div>
        `;
        
        // Render query attribution heatmap
        const queryTokensHtml = queryAttrs.map(([token, score]) => {
            const bgColor = colorMap[score] || 'rgb(247,247,247)';
            const textColor = this.getContrastTextColor(bgColor);
            return `
                <span 
                    class="inline-block px-2 py-1 m-0.5 rounded text-sm font-mono border border-gray-300 cursor-help transition-transform hover:scale-105"
                    style="background-color: ${bgColor}; color: ${textColor};"
                    onmouseenter="window.contentManager.showSHAPEncoderTooltip(event, '${this.escapeHtml(token)}', ${score})"
                    onmouseleave="window.contentManager.hideSHAPEncoderTooltip()"
                >${this.escapeHtml(token)}</span>
            `;
        }).join('');
        
        const queryHtml = `
            <div class="mb-4">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="font-medium text-gray-800 flex items-center">
                        <i class="fas fa-file-text mr-2 text-blue-500"></i>
                        Query
                        <span class="ml-2 text-xs text-gray-500">(Hover/click tokens to see which inputs influenced it)</span>
                    </h5>
                    <span class="text-xs text-gray-500">${queryAttrs.length} tokens</span>
                </div>
                <div class="bg-white rounded-lg p-3 border border-gray-200">
                    ${queryTokensHtml}
                </div>
            </div>
        `;
        
        // Render document attribution heatmap
        const docTokensHtml = docAttrs.map(([token, score]) => {
            const bgColor = colorMap[score] || 'rgb(247,247,247)';
            const textColor = this.getContrastTextColor(bgColor);
            return `
                <span 
                    class="inline-block px-2 py-1 m-0.5 rounded text-sm font-mono border border-gray-300 cursor-help transition-transform hover:scale-105"
                    style="background-color: ${bgColor}; color: ${textColor};"
                    onmouseenter="window.contentManager.showSHAPEncoderTooltip(event, '${this.escapeHtml(token)}', ${score})"
                    onmouseleave="window.contentManager.hideSHAPEncoderTooltip()"
                >${this.escapeHtml(token)}</span>
            `;
        }).join('');
        
        const docHtml = `
            <div class="mb-4">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="font-medium text-gray-800 flex items-center">
                        <i class="fas fa-file-alt mr-2 text-green-500"></i>
                        Document
                        <span class="ml-2 text-xs text-gray-500">(Hover/click tokens to see which inputs influenced it)</span>
                    </h5>
                    <span class="text-xs text-gray-500">${docAttrs.length} tokens</span>
                </div>
                <div class="bg-white rounded-lg p-3 border border-gray-200">
                    ${docTokensHtml}
                </div>
            </div>
        `;
        
        // Combine all HTML
        contentArea.innerHTML = contextHtml + queryHtml + docHtml;
    }
    
    showSHAPEncoderTooltip(event, token, score) {
        // Remove any existing tooltip
        this.hideSHAPEncoderTooltip();
        
        const tooltip = document.createElement('div');
        tooltip.id = 'shap-encoder-tooltip';
        tooltip.className = 'absolute z-50 bg-gray-900 text-white text-xs rounded py-2 px-3 pointer-events-none shadow-lg';
        tooltip.innerHTML = `
            <div class="font-bold mb-1">${this.escapeHtml(token)}</div>
            <div>Attribution: <span class="font-mono">${score.toFixed(4)}</span></div>
            <div class="text-gray-300 mt-1 text-xs">
                ${score > 0 ? '↑ Increases score' : score < 0 ? '↓ Decreases score' : 'Neutral'}
            </div>
        `;
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = event.target.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltip.offsetWidth / 2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
        
        // Ensure tooltip stays within viewport
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipRect.left < 0) {
            tooltip.style.left = '8px';
        } else if (tooltipRect.right > window.innerWidth) {
            tooltip.style.left = `${window.innerWidth - tooltipRect.width - 8}px`;
        }
        
        if (tooltipRect.top < 0) {
            tooltip.style.top = `${rect.bottom + 8}px`;
        }
    }
    
    hideSHAPEncoderTooltip() {
        const tooltip = document.getElementById('shap-encoder-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    // ==================== End SHAP Encoder Methods ====================

    // ==================== IG Encoder Visualization Methods ====================
    
    renderIGEncoderVisualization() {
        const container = document.getElementById('ig-encoder-visualization-container');
        if (!container || !this.currentIGEncoderData || this.currentIGEncoderData.length === 0) {
            return;
        }
        
        // Reset to first sample and document
        this.currentIGEncoderSampleIndex = 0;
        const firstSample = this.currentIGEncoderData[0];
        this.totalIGEncoderDocs = firstSample.attributions ? firstSample.attributions.length : 0;
        this.currentIGEncoderDocIndex = 0;
        
        // Create HTML structure
        container.innerHTML = `
            <!-- Sample Navigation (Query Level) -->
            <div class="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg border border-gray-300">
                <div class="flex items-center space-x-2">
                    <span class="text-sm text-gray-600 font-medium">Query:</span>
                    <span id="ig-encoder-sample-counter" class="text-sm text-gray-600">1 of ${this.totalIGEncoderSamples}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <button 
                        id="ig-encoder-prev-sample-btn"
                        onclick="window.contentManager.navigateIGEncoderSample(-1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <i class="fas fa-chevron-left mr-1"></i>Previous Query
                    </button>
                    <button 
                        id="ig-encoder-next-sample-btn"
                        onclick="window.contentManager.navigateIGEncoderSample(1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Next Query<i class="fas fa-chevron-right ml-1"></i>
                    </button>
                </div>
            </div>
            
            <!-- Document Navigation -->
            <div class="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg border border-gray-300">
                <div class="flex items-center space-x-2">
                    <span class="text-sm text-gray-600 font-medium">Document:</span>
                    <span id="ig-encoder-doc-counter" class="text-sm text-gray-600">1 of ${this.totalIGEncoderDocs}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <button 
                        id="ig-encoder-prev-doc-btn"
                        onclick="window.contentManager.navigateIGEncoderDocument(-1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <i class="fas fa-chevron-left mr-1"></i>Previous Document
                    </button>
                    <button 
                        id="ig-encoder-next-doc-btn"
                        onclick="window.contentManager.navigateIGEncoderDocument(1)"
                        class="px-3 py-2 bg-white border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Next Document<i class="fas fa-chevron-right ml-1"></i>
                    </button>
                </div>
            </div>
            
            <!-- Visualization Area -->
            <div id="ig-encoder-content-area"></div>
        `;
        
        // Render the first view
        this.renderCurrentIGEncoderView();
    }
    
    navigateIGEncoderSample(direction) {
        const newIndex = this.currentIGEncoderSampleIndex + direction;
        if (newIndex >= 0 && newIndex < this.totalIGEncoderSamples) {
            this.currentIGEncoderSampleIndex = newIndex;
            
            // Reset document index and update total docs for new sample
            const currentSample = this.currentIGEncoderData[this.currentIGEncoderSampleIndex];
            this.totalIGEncoderDocs = currentSample.attributions ? currentSample.attributions.length : 0;
            this.currentIGEncoderDocIndex = 0;
            
            this.renderCurrentIGEncoderView();
        }
    }
    
    navigateIGEncoderDocument(direction) {
        const newIndex = this.currentIGEncoderDocIndex + direction;
        if (newIndex >= 0 && newIndex < this.totalIGEncoderDocs) {
            this.currentIGEncoderDocIndex = newIndex;
            this.renderCurrentIGEncoderView();
        }
    }
    
    renderCurrentIGEncoderView() {
        const contentArea = document.getElementById('ig-encoder-content-area');
        if (!contentArea) return;
        
        // Update navigation counters
        const sampleCounter = document.getElementById('ig-encoder-sample-counter');
        const docCounter = document.getElementById('ig-encoder-doc-counter');
        
        if (sampleCounter) {
            sampleCounter.textContent = `${this.currentIGEncoderSampleIndex + 1} of ${this.totalIGEncoderSamples}`;
        }
        
        if (docCounter) {
            docCounter.textContent = `${this.currentIGEncoderDocIndex + 1} of ${this.totalIGEncoderDocs}`;
        }
        
        // Update button states
        const prevSampleBtn = document.getElementById('ig-encoder-prev-sample-btn');
        const nextSampleBtn = document.getElementById('ig-encoder-next-sample-btn');
        const prevDocBtn = document.getElementById('ig-encoder-prev-doc-btn');
        const nextDocBtn = document.getElementById('ig-encoder-next-doc-btn');
        
        if (prevSampleBtn) {
            prevSampleBtn.disabled = this.currentIGEncoderSampleIndex === 0;
            prevSampleBtn.classList.toggle('opacity-50', prevSampleBtn.disabled);
            prevSampleBtn.classList.toggle('cursor-not-allowed', prevSampleBtn.disabled);
        }
        
        if (nextSampleBtn) {
            nextSampleBtn.disabled = this.currentIGEncoderSampleIndex >= this.totalIGEncoderSamples - 1;
            nextSampleBtn.classList.toggle('opacity-50', nextSampleBtn.disabled);
            nextSampleBtn.classList.toggle('cursor-not-allowed', nextSampleBtn.disabled);
        }
        
        if (prevDocBtn) {
            prevDocBtn.disabled = this.currentIGEncoderDocIndex === 0;
            prevDocBtn.classList.toggle('opacity-50', prevDocBtn.disabled);
            prevDocBtn.classList.toggle('cursor-not-allowed', prevDocBtn.disabled);
        }
        
        if (nextDocBtn) {
            nextDocBtn.disabled = this.currentIGEncoderDocIndex >= this.totalIGEncoderDocs - 1;
            nextDocBtn.classList.toggle('opacity-50', nextDocBtn.disabled);
            nextDocBtn.classList.toggle('cursor-not-allowed', nextDocBtn.disabled);
        }
        
        // Get current data
        const currentSample = this.currentIGEncoderData[this.currentIGEncoderSampleIndex];
        if (!currentSample || !currentSample.attributions || currentSample.attributions.length === 0) {
            contentArea.innerHTML = '<div class="text-gray-500 text-center py-8">No attribution data available</div>';
            return;
        }
        
        const currentDoc = currentSample.attributions[this.currentIGEncoderDocIndex];
        const queryAttrs = currentDoc.query_attributions || [];
        const docAttrs = currentDoc.document_attributions || [];
        
        // Collect all scores for colormap generation
        const allScores = [
            ...queryAttrs.map(attr => attr[1]),
            ...docAttrs.map(attr => attr[1])
        ];
        const colorMap = this.generateMatplotlibColorMap(allScores, 'RdBu_r');
        
        // Render context panel with convergence deltas
        const contextHtml = `
            <div class="bg-purple-50 rounded-lg p-4 mb-4 border border-gray-300">
                <div class="grid grid-cols-3 gap-4 text-sm mb-3">
                    <div>
                        <span class="font-medium text-gray-700">Query ID:</span>
                        <span class="ml-2 text-gray-900 font-mono">${currentSample.query_id || 'N/A'}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Document ID:</span>
                        <span class="ml-2 text-gray-900 font-mono">${currentDoc.document_id || 'N/A'}</span>
                        <i class="fas fa-info-circle ml-1 text-gray-400 cursor-help" 
                            onmouseenter="contentManager.showConvergenceDeltaTooltip(event, ${currentDoc.query_convergence_delta}, ${currentDoc.document_convergence_delta})" 
                            onmouseleave="contentManager.hideConvergenceDeltaTooltip()" 
                            title="Convergence Delta"></i>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Rerank Score:</span>
                        <span class="ml-2 text-gray-900 font-bold">${currentDoc.rerank_score?.toFixed(4) || 'N/A'}</span>
                    </div>
                </div>
                <div class="mt-3 pt-3 border-t border-gray-200">
                    <div class="font-medium text-gray-700 mb-1">Query Text:</div>
                    <div class="text-gray-900 bg-white rounded p-2 text-sm border border-gray-300">${this.escapeHtml(currentSample.query)}</div>
                </div>
                <div class="mt-2">
                    <button 
                        onclick="showDocumentModal('${this.escapeHtml(currentDoc.document_text).replace(/'/g, '&apos;')}', '${currentDoc.document_id || 'N/A'}')"
                        class="text-purple-600 hover:text-purple-800 text-sm font-medium"
                    >
                        <i class="fas fa-file-alt mr-1"></i>View Full Document
                    </button>
                </div>
            </div>
        `;
        
        // Render query attribution heatmap
        const queryTokensHtml = queryAttrs.map(([token, score]) => {
            const bgColor = colorMap[score] || 'rgb(247,247,247)';
            const textColor = this.getContrastTextColor(bgColor);
            return `
                <span 
                    class="inline-block px-2 py-1 m-0.5 rounded text-sm font-mono border border-gray-300 cursor-help transition-transform hover:scale-105"
                    style="background-color: ${bgColor}; color: ${textColor};"
                    onmouseenter="window.contentManager.showIGEncoderTooltip(event, '${this.escapeHtml(token)}', ${score})"
                    onmouseleave="window.contentManager.hideIGEncoderTooltip()"
                >${this.escapeHtml(token)}</span>
            `;
        }).join('');
        
        const queryHtml = `
            <div class="mb-4">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="font-medium text-gray-800 flex items-center">
                        <i class="fas fa-file-alt mr-2 text-blue-500"></i>
                        Query
                        <span class="ml-2 text-xs text-gray-500">(Hover/click tokens to see which inputs influenced it)</span>
                    </h5>
                    <span class="text-xs text-gray-500">${queryAttrs.length} tokens</span>
                </div>
                <div class="bg-white rounded-lg p-3 border border-gray-200">
                    ${queryTokensHtml}
                </div>
            </div>
        `;
        
        // Render document attribution heatmap
        const docTokensHtml = docAttrs.map(([token, score]) => {
            const bgColor = colorMap[score] || 'rgb(247,247,247)';
            const textColor = this.getContrastTextColor(bgColor);
            return `
                <span 
                    class="inline-block px-2 py-1 m-0.5 rounded text-sm font-mono border border-gray-300 cursor-help transition-transform hover:scale-105"
                    style="background-color: ${bgColor}; color: ${textColor};"
                    onmouseenter="window.contentManager.showIGEncoderTooltip(event, '${this.escapeHtml(token)}', ${score})"
                    onmouseleave="window.contentManager.hideIGEncoderTooltip()"
                >${this.escapeHtml(token)}</span>
            `;
        }).join('');
        
        const docHtml = `
            <div class="mb-4">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="font-medium text-gray-800 flex items-center">
                        <i class="fas fa-file-alt mr-2 text-green-500"></i>
                        Document
                        <span class="ml-2 text-xs text-gray-500">(Hover/click tokens to see which inputs influenced it)</span>
                    </h5>
                    <span class="text-xs text-gray-500">${docAttrs.length} tokens</span>
                </div>
                <div class="bg-white rounded-lg p-3 border border-gray-200">
                    ${docTokensHtml}
                </div>
            </div>
        `;
        
        // Combine all HTML
        contentArea.innerHTML = contextHtml + queryHtml + docHtml;
    }
    
    showConvergenceDeltaTooltip(event, queryDelta, docDelta) {
        this.hideConvergenceDeltaTooltip();
        
        const tooltip = document.createElement('div');
        tooltip.id = 'ig-encoder-convergence-tooltip';
        tooltip.className = 'absolute z-50 bg-gray-900 text-white text-xs rounded py-2 px-3 pointer-events-none shadow-lg';
        
        const formatDelta = (delta, label) => {
            if (delta === undefined || delta === null) {
                return `<div class="text-gray-400">${label}: N/A</div>`;
            }
            const threshold = 0.05;
            const colorClass = Math.abs(delta) < threshold ? 'text-green-400' : (Math.abs(delta) < 0.1 ? 'text-orange-400' : 'text-red-400');
            const icon = Math.abs(delta) < threshold ? 'fa-check-circle' : (Math.abs(delta) < 0.1 ? 'fa-exclamation-triangle' : 'fa-times-circle');
            return `<div class="flex items-center ${colorClass}"><i class="fas ${icon} mr-1"></i>${label} δ: ${delta.toFixed(4)}</div>`;
        };
        
        tooltip.innerHTML = `
            <div class="font-bold mb-1">Convergence Delta</div>
            ${formatDelta(queryDelta, 'Query')}
            ${formatDelta(docDelta, 'Document')}
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = event.target.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltip.offsetWidth / 2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
        
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipRect.left < 0) {
            tooltip.style.left = '8px';
        } else if (tooltipRect.right > window.innerWidth) {
            tooltip.style.left = `${window.innerWidth - tooltipRect.width - 8}px`;
        }
        if (tooltipRect.top < 0) {
            tooltip.style.top = `${rect.bottom + 8}px`;
        }
    }
    
    hideConvergenceDeltaTooltip() {
        const tooltip = document.getElementById('ig-encoder-convergence-tooltip');
        if (tooltip) tooltip.remove();
    }
    
    showIGEncoderTooltip(event, token, score) {
        // Remove any existing tooltip
        this.hideIGEncoderTooltip();
        
        const tooltip = document.createElement('div');
        tooltip.id = 'ig-encoder-tooltip';
        tooltip.className = 'absolute z-50 bg-gray-900 text-white text-xs rounded py-2 px-3 pointer-events-none shadow-lg';
        tooltip.innerHTML = `
            <div class="font-bold mb-1">${this.escapeHtml(token)}</div>
            <div>Attribution: <span class="font-mono">${score.toFixed(4)}</span></div>
            <div class="text-gray-300 mt-1 text-xs">
                ${score > 0 ? '↑ Increases score' : score < 0 ? '↓ Decreases score' : 'Neutral'}
            </div>
        `;
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = event.target.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltip.offsetWidth / 2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
        
        // Ensure tooltip stays within viewport
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipRect.left < 0) {
            tooltip.style.left = '8px';
        } else if (tooltipRect.right > window.innerWidth) {
            tooltip.style.left = `${window.innerWidth - tooltipRect.width - 8}px`;
        }
        
        if (tooltipRect.top < 0) {
            tooltip.style.top = `${rect.bottom + 8}px`;
        }
    }
    
    hideIGEncoderTooltip() {
        const tooltip = document.getElementById('ig-encoder-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    // ==================== End IG Encoder Methods ====================


    generateMatplotlibColorMap(scores, colormapName = 'RdBu_r') {
        // Use logarithmic scaling for better visual distinction across the full range
        // This helps differentiate between values like 1.1 and 5.2 more clearly
        if (!scores || scores.length === 0) return {};
        
        // Find absolute max for symmetric scaling
        const maxAbsScore = Math.max(...scores.map(Math.abs));
        if (maxAbsScore === 0) return {};
        
        const colorMap = {};
        
        for (const score of scores) {
            const absScore = Math.abs(score);
            const isPositive = score >= 0;
            
            // Apply logarithmic scaling
            const logScore = Math.log1p(absScore); // log(1 + abs(score))
            const maxLogScore = Math.log1p(maxAbsScore);
            const normalized = logScore / maxLogScore;
            
            // Apply a power function to spread out mid-range values
            const spread = Math.pow(normalized, 0.7);
            
            // Get color based on spread magnitude and sign
            colorMap[score] = this.getMatplotlibColorByMagnitude(spread, isPositive, colormapName);
        }
        
        return colorMap;
    }
    
    getMatplotlibColorByMagnitude(magnitude, isPositive, colormapName) {
        // Clamp magnitude to [0, 1] range
        const clampedMagnitude = Math.max(0, Math.min(1, magnitude));
        
        let t;
        
        if (colormapName === 'RdBu_r') {
            // For RdBu_r: 0 = red (negative), 0.5 = white (neutral), 1 = blue (positive)
            // Map based on magnitude and sign
            if (isPositive) {
                // Positive values: interpolate from neutral (0.5) to blue (1.0)
                t = 0.5 + (clampedMagnitude * 0.5);
            } else {
                // Negative values: interpolate from neutral (0.5) to red (0.0)
                t = 0.5 - (clampedMagnitude * 0.5);
            }
        } else {
            // For other colormaps, use magnitude directly
            t = clampedMagnitude;
        }
        
        return this.getMatplotlibColor(t, colormapName);
    }

    getMatplotlibColor(normalizedValue, colormapName) {
        // Clamp value to [0, 1]
        const t = Math.max(0, Math.min(1, normalizedValue));
        
        switch (colormapName) {
            case 'RdBu_r': // Red-Blue reversed (red for negative, blue for positive)
                return this.interpolateRdBu_r(t);
            case 'RdYlBu_r': // Red-Yellow-Blue reversed
                return this.interpolateRdYlBu_r(t);
            case 'viridis': // Perceptually uniform colormap
                return this.interpolateViridis(t);
            case 'plasma':
                return this.interpolatePlasma(t);
            default:
                return this.interpolateRdBu_r(t);
        }
    }

    interpolateRdBu_r(t) {
        // RdBu_r colormap: Red (low) -> White (middle) -> Blue (high)
        // Made more subtle with lighter colors and wider neutral zone
        if (t < 0.35) {
            // Strong negative: light red to very light red
            const localT = t / 0.35; // Map [0, 0.35] to [0, 1]
            const r = Math.round(239 + (252 - 239) * localT);
            const g = Math.round(138 + (220 - 138) * localT);
            const b = Math.round(138 + (220 - 138) * localT);
            return `rgb(${r}, ${g}, ${b})`;
        } else if (t < 0.5) {
            // Weak negative: very light red to white
            const localT = (t - 0.35) / 0.15; // Map [0.35, 0.5] to [0, 1]
            const r = Math.round(252 + (255 - 252) * localT);
            const g = Math.round(220 + (255 - 220) * localT);
            const b = Math.round(220 + (255 - 220) * localT);
            return `rgb(${r}, ${g}, ${b})`;
        } else if (t < 0.65) {
            // Weak positive: white to very light blue
            const localT = (t - 0.5) / 0.15; // Map [0.5, 0.65] to [0, 1]
            const r = Math.round(255 - (255 - 220) * localT);
            const g = Math.round(255 - (255 - 230) * localT);
            const b = Math.round(255 - (255 - 252) * localT);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Strong positive: very light blue to light blue
            const localT = (t - 0.65) / 0.35; // Map [0.65, 1] to [0, 1]
            const r = Math.round(220 - (220 - 158) * localT);
            const g = Math.round(230 - (230 - 188) * localT);
            const b = Math.round(252 - (252 - 235) * localT);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    interpolateRdYlBu_r(t) {
        // RdYlBu_r colormap: Red -> Yellow -> Blue
        if (t < 0.5) {
            // Red to Yellow
            const localT = t * 2;
            const r = Math.round(215 + (255 - 215) * localT);
            const g = Math.round(48 + (255 - 48) * localT);
            const b = Math.round(39 + (224 - 39) * localT);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Yellow to Blue
            const localT = (t - 0.5) * 2;
            const r = Math.round(255 - (255 - 69) * localT);
            const g = Math.round(255 - (255 - 117) * localT);
            const b = Math.round(224 - (224 - 180) * localT);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    interpolateViridis(t) {
        // Viridis colormap approximation
        const colors = [
            [68, 1, 84],    // Dark purple
            [59, 82, 139],  // Blue
            [33, 144, 140], // Teal
            [92, 200, 99],  // Green
            [253, 231, 37]  // Yellow
        ];
        
        const scaledT = t * (colors.length - 1);
        const index = Math.floor(scaledT);
        const fraction = scaledT - index;
        
        if (index >= colors.length - 1) {
            const [r, g, b] = colors[colors.length - 1];
            return `rgb(${r}, ${g}, ${b})`;
        }
        
        const [r1, g1, b1] = colors[index];
        const [r2, g2, b2] = colors[index + 1];
        
        const r = Math.round(r1 + (r2 - r1) * fraction);
        const g = Math.round(g1 + (g2 - g1) * fraction);
        const b = Math.round(b1 + (b2 - b1) * fraction);
        
        return `rgb(${r}, ${g}, ${b})`;
    }

    interpolatePlasma(t) {
        // Plasma colormap approximation
        const colors = [
            [13, 8, 135],    // Dark blue
            [84, 2, 163],    // Purple
            [139, 10, 165],  // Magenta
            [185, 50, 137],  // Pink
            [219, 92, 104],  // Red
            [244, 136, 73],  // Orange
            [254, 188, 43],  // Yellow
            [240, 249, 33]   // Light yellow
        ];
        
        const scaledT = t * (colors.length - 1);
        const index = Math.floor(scaledT);
        const fraction = scaledT - index;
        
        if (index >= colors.length - 1) {
            const [r, g, b] = colors[colors.length - 1];
            return `rgb(${r}, ${g}, ${b})`;
        }
        
        const [r1, g1, b1] = colors[index];
        const [r2, g2, b2] = colors[index + 1];
        
        const r = Math.round(r1 + (r2 - r1) * fraction);
        const g = Math.round(g1 + (g2 - g1) * fraction);
        const b = Math.round(b1 + (b2 - b1) * fraction);
        
        return `rgb(${r}, ${g}, ${b})`;
    }

    getContrastTextColor(backgroundColor) {
        // Convert RGB to luminance and determine if we need light or dark text
        let rgb = backgroundColor.match(/\d+/g);
        if (!rgb || rgb.length < 3) return '#000000';
        
        rgb = rgb.map(Number);
        const [r, g, b] = rgb;
        
        // Calculate relative luminance
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        
        // Use dark text for light backgrounds, light text for dark backgrounds
        return luminance > 0.5 ? '#000000' : '#ffffff';
    }

    getTokenColor(score, minScore, maxScore, scoreRange) {
        // Normalize score to [-1, 1] range if possible, otherwise use relative scaling
        let normalizedScore;
        if (scoreRange === 0) {
            normalizedScore = 0; // All scores are the same
        } else if (minScore >= 0) {
            // All positive scores
            normalizedScore = score / maxScore;
        } else if (maxScore <= 0) {
            // All negative scores
            normalizedScore = score / Math.abs(minScore);
        } else {
            // Mixed positive and negative scores
            normalizedScore = score / Math.max(Math.abs(minScore), Math.abs(maxScore));
        }
        
        // Clamp to [-1, 1]
        normalizedScore = Math.max(-1, Math.min(1, normalizedScore));
        
        let bgColor, textColor;
        
        if (Math.abs(normalizedScore) < 0.05) {
            // Neutral (very close to zero)
            bgColor = '#f3f4f6'; // gray-100
            textColor = '#374151'; // gray-700
        } else if (normalizedScore > 0) {
            // Positive scores - green scale
            const intensity = Math.abs(normalizedScore);
            if (intensity < 0.33) {
                bgColor = '#dcfce7'; // green-100
                textColor = '#166534'; // green-800
            } else if (intensity < 0.66) {
                bgColor = '#bbf7d0'; // green-200
                textColor = '#14532d'; // green-900
            } else {
                bgColor = '#86efac'; // green-300
                textColor = '#14532d'; // green-900
            }
        } else {
            // Negative scores - red scale
            const intensity = Math.abs(normalizedScore);
            if (intensity < 0.33) {
                bgColor = '#fee2e2'; // red-100
                textColor = '#991b1b'; // red-800
            } else if (intensity < 0.66) {
                bgColor = '#fecaca'; // red-200
                textColor = '#7f1d1d'; // red-900
            } else {
                bgColor = '#fca5a5'; // red-300
                textColor = '#7f1d1d'; // red-900
            }
        }
        
        return { bg: bgColor, text: textColor };
    }

    populateLogsTab(job) {
        const logsElement = document.getElementById('job-logs');
        
        // Get live job data from queue manager if available (for real-time logs)
        const queueJob = window.queueManager ? window.queueManager.jobs.find(j => j.id === job.id) : null;
        const liveJob = queueJob || job; // Prefer queue manager data for live logs
        const hasNewLogs = queueJob && queueJob.hasNewLogs;
        
        // Add/update new logs indicator in Logs tab
        const logsTab = document.querySelector('.job-detail-tab[data-tab="logs"]');
        if (logsTab) {
            let indicator = logsTab.querySelector('.new-logs-indicator');
            if (hasNewLogs) {
                if (!indicator) {
                    indicator = document.createElement('span');
                    indicator.className = 'new-logs-indicator inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse ml-2';
                    indicator.title = 'New logs available';
                    logsTab.appendChild(indicator);
                }
            } else if (indicator) {
                indicator.remove();
            }
        }
        
        if (liveJob.log && liveJob.log.length > 0) {
            // Combine all logs into a single chronological view using live data
            const allLogs = liveJob.log.map(logEntry => {
                // Format the log entry with better styling
                const timestamp = logEntry.timestamp || new Date().toISOString();
                const level = logEntry.level || 'INFO';
                const message = logEntry.message || 'No message';
                
                return `[${timestamp}] ${level}: ${message}`;
            }).join('\n');
            
            logsElement.textContent = allLogs;
        } else {
            // Fallback when no structured logs are available
            const fallbackMessage = liveJob.status === 'FAILED' 
                ? `[${new Date().toISOString()}] ERROR: Job #${liveJob.id} failed - detailed logs not available\n[${new Date().toISOString()}] INFO: Status: ${liveJob.status}\n[${new Date().toISOString()}] INFO: Message: ${liveJob.status_message || 'No error message available'}`
                : liveJob.status === 'COMPLETED' || liveJob.status === 'SUCCESS'
                    ? `[${new Date().toISOString()}] INFO: Job #${liveJob.id} completed successfully\n[${new Date().toISOString()}] INFO: Runtime: ${liveJob.runtime ? liveJob.runtime.toFixed(2) + 's' : 'Unknown'}\n[${new Date().toISOString()}] INFO: Status: ${liveJob.status}\n[${new Date().toISOString()}] INFO: Detailed logs not available for this job`
                    : `[${new Date().toISOString()}] INFO: Job #${liveJob.id} status: ${liveJob.status}\n[${new Date().toISOString()}] INFO: Logs will be available when job completes`;

            logsElement.textContent = fallbackMessage;
        }
    }

    populateTagsTab(job) {
        // Store current job for tag operations
        this.currentTagJob = job;
        
        // Get the method ID from the current content or from the job
        // If we're viewing a method, use that ID; otherwise try to get it from the job
        this.currentTagMethodId = job.method_id || (this.currentContent && this.currentContent.type === 'method' ? this.currentContent.id : null);
        
        // Populate available tags dropdown
        this.loadAvailableTags();
        
        // Load and display current tags for this job
        this.loadCurrentTags(job.id);
        
        // Setup tag preview functionality
        this.setupTagPreview();
    }
    
    // Method called by QueueManager when new logs arrive
    notifyNewLogs(jobId) {
        // Only update if the modal is showing this job
        if (this.currentJobInModal !== jobId) return;
        
        // Add indicator to Logs tab if not already present
        const logsTab = document.querySelector('.job-detail-tab[data-tab="logs"]');
        if (logsTab && !logsTab.querySelector('.new-logs-indicator')) {
            const indicator = document.createElement('span');
            indicator.className = 'new-logs-indicator inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse ml-2';
            indicator.title = 'New logs available';
            logsTab.appendChild(indicator);
        }
        
        // If logs tab is currently active, refresh the logs with live data
        const logsTabContent = document.getElementById('logs-tab');
        if (logsTabContent && !logsTabContent.classList.contains('hidden')) {
            const job = this.findJobById(jobId);
            if (job) {
                // Get current scroll position
                const logsElement = document.getElementById('job-logs');
                const wasAtBottom = logsElement.scrollHeight - logsElement.scrollTop <= logsElement.clientHeight + 50;
                
                this.populateLogsTab(job);
                
                // Auto-scroll to bottom if user was already at the bottom
                if (wasAtBottom) {
                    setTimeout(() => {
                        logsElement.scrollTop = logsElement.scrollHeight;
                    }, 0);
                }
                
                // Clear new logs indicator since we just showed them
                this.clearNewLogsIndicator(jobId);
            }
        }
    }
    
    // Clear new logs indicator
    clearNewLogsIndicator(jobId) {
        const logsTab = document.querySelector('.job-detail-tab[data-tab="logs"]');
        if (logsTab) {
            const indicator = logsTab.querySelector('.new-logs-indicator');
            if (indicator) {
                indicator.remove();
            }
        }
        
        // Clear in queue manager too
        if (window.queueManager) {
            const queueJob = window.queueManager.jobs.find(j => j.id === jobId);
            if (queueJob) {
                queueJob.hasNewLogs = false;
                window.queueManager.saveToStorage();
                window.queueManager.updateTopbarUI();
            }
        }
    }

    async loadAvailableTags() {
        try {
            const response = await fetch('/api/tags/available/');
            if (!response.ok) throw new Error('Failed to load available tags');
            
            const availableTags = await response.json();
            const select = document.getElementById('available-tags-select');
            
            // Clear existing options except the first one
            select.innerHTML = '<option value="">Select a tag...</option>';
            
            // Add available tags to dropdown
            Object.keys(availableTags).forEach(tagName => {
                const tag = availableTags[tagName];
                const option = document.createElement('option');
                option.value = tagName;
                option.textContent = `${tag.name} (${tag.category || 'General'})`;
                option.dataset.tagData = JSON.stringify(tag);
                select.appendChild(option);
            });
            
        } catch (error) {
            console.error('Error loading available tags:', error);
            // Show error state
            const select = document.getElementById('available-tags-select');
            select.innerHTML = '<option value="">Error loading tags...</option>';
        }
    }

    async loadCurrentTags(jobId) {
        try {
            const response = await fetch(`/api/tags/job/?job_id=${jobId}`);
            if (!response.ok) throw new Error('Failed to load current tags');
            
            const currentTags = await response.json();
            this.displayCurrentTags(currentTags);
            
            // Update the job data in testsData with the latest tags
            await this.updateJobTags(jobId, currentTags);
            
        } catch (error) {
            console.error('Error loading current tags:', error);
            this.displayCurrentTags([]);
        }
    }

    async updateJobTags(jobId, tags) {
        // Find and update the job with new tag data
        for (const test of this.testsData) {
            for (const method of test.methods || []) {
                if (method.jobs) {
                    const job = method.jobs.find(j => j.id === jobId);
                    if (job) {
                        job.tags = tags;
                        // If this is the current method being displayed, refresh the job cards
                        // This will update the tag icons immediately, even if the modal is open
                        if (this.currentContent && this.currentContent.type === 'method' && this.currentContent.id === method.id) {
                            await this.populateJobsTab(method);
                        }
                        return;
                    }
                }
            }
        }
    }

    async loadTagsForJobs(jobs) {
        // Load tags for all jobs in parallel
        const tagPromises = jobs.map(async (job) => {
            try {
                const response = await fetch(`/api/tags/job/?job_id=${job.id}`);
                if (response.ok) {
                    const tags = await response.json();
                    job.tags = tags;
                }
            } catch (error) {
                console.warn(`Failed to load tags for job ${job.id}:`, error);
                job.tags = []; // Set empty array as fallback
            }
        });

        await Promise.all(tagPromises);
    }

    displayCurrentTags(tags) {
        const noTagsMessage = document.getElementById('no-tags-message');
        const tagsList = document.getElementById('tags-list');
        const tagsCount = document.getElementById('tags-count');
        
        tagsCount.textContent = tags.length;
        
        if (tags.length === 0) {
            noTagsMessage.classList.remove('hidden');
            tagsList.classList.add('hidden');
            return;
        }
        
        noTagsMessage.classList.add('hidden');
        tagsList.classList.remove('hidden');
        
        // Generate HTML for current tags
        tagsList.innerHTML = tags.map(tag => `
            <div class="border border-gray-200 rounded-lg p-4 bg-white">
                <div class="flex items-start justify-between mb-3">
                    <div class="flex-1">
                        <div class="flex items-center space-x-2 mb-1">
                            <h5 class="font-medium text-gray-800">${this.escapeHtml(tag.name)}</h5>
                            ${tag.category ? `<span class="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">${this.escapeHtml(tag.category)}</span>` : ''}
                        </div>
                        <p class="text-sm text-gray-600 mb-2">${this.escapeHtml(tag.description || 'No description available')}</p>
                        ${tag.paper_reference ? `<p class="text-xs text-blue-600 mb-2"><i class="fas fa-external-link-alt mr-1"></i>Reference: <a href="${tag.paper_reference}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">${this.escapeHtml(tag.paper_reference)}</a></p>` : ''}
                    </div>
                    <button class="text-red-500 hover:text-red-700 p-1" onclick="removeTagFromJob('${tag.id}')" title="Remove tag">
                        <i class="fas fa-trash text-sm"></i>
                    </button>
                </div>
                
                <!-- Comment Section -->
                <div class="border-t border-gray-100 pt-3">
                    <div class="flex items-center justify-between mb-2">
                        <label class="block text-sm font-medium text-gray-700">Comment:</label>
                        <span id="comment-status-${tag.id}" class="text-xs font-medium hidden transition-all duration-300"></span>
                    </div>
                    
                    <textarea 
                        id="tag-comment-${tag.id}" 
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent text-sm"
                        rows="3"
                        placeholder="Add your comments about this tag..."
                        onchange="updateTagComment('${tag.id}', this.value)"
                    >${this.escapeHtml(tag.comment || '')}</textarea>
                </div>
            </div>
        `).join('');
    }

    setupTagPreview() {
        const select = document.getElementById('available-tags-select');
        const addBtn = document.getElementById('add-tag-btn');
        const preview = document.getElementById('tag-preview');
        
        select.addEventListener('change', (e) => {
            const selectedOption = e.target.selectedOptions[0];
            if (selectedOption && selectedOption.dataset.tagData) {
                const tagData = JSON.parse(selectedOption.dataset.tagData);
                this.showTagPreview(tagData);
                addBtn.disabled = false;
            } else {
                this.hideTagPreview();
                addBtn.disabled = true;
            }
        });
    }

    showTagPreview(tag) {
        const preview = document.getElementById('tag-preview');
        preview.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="flex-shrink-0">
                    <i class="fas fa-tag text-blue-500 mt-1"></i>
                </div>
                <div class="flex-1">
                    <div class="flex items-center space-x-2 mb-1">
                        <h6 class="font-medium text-gray-800">${this.escapeHtml(tag.name)}</h6>
                        ${tag.category ? `<span class="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">${this.escapeHtml(tag.category)}</span>` : ''}
                    </div>
                    <p class="text-sm text-gray-600 mb-2">${this.escapeHtml(tag.description || 'No description available')}</p>
                    ${tag.paper_reference ? `<p class="text-xs text-blue-600 mb-1"><i class="fas fa-external-link-alt mr-1"></i>Reference: <a href="${tag.paper_reference}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline">${this.escapeHtml(tag.paper_reference)}</a></p>` : ''}
                </div>
            </div>
        `;
        preview.classList.remove('hidden');
    }

    hideTagPreview() {
        const preview = document.getElementById('tag-preview');
        preview.classList.add('hidden');
    }

    populateActivityHistory(methodData) {
        const activityContainer = document.getElementById('activity-history-container');
        if (!activityContainer) return;

        const activities = methodData.activities || [];
        
        if (activities.length === 0) {
            activityContainer.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-clock text-gray-300 text-2xl mb-2"></i>
                    <p class="text-gray-500">No activity recorded yet</p>
                </div>
            `;
            return;
        }

        let activitiesHtml = '';
        activities.forEach(activity => {
            const icon = this.getActivityIcon(activity.activity_type);
            const bgColor = this.getActivityBgColor(activity.activity_type);
            const iconColor = this.getActivityIconColor(activity.activity_type);
            const timestamp = this.formatTimestamp(activity.timestamp);

            activitiesHtml += `
                <div class="flex items-center space-x-3 p-3 bg-white rounded-lg border border-gray-200 hover:border-gray-300 cursor-pointer transition-colors duration-200" onclick="openActivityModal('${activity.id}')">
                    <div class="flex-shrink-0 w-8 h-8 ${bgColor} rounded-full flex items-center justify-center">
                        <i class="${icon} ${iconColor} text-xs"></i>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-medium text-gray-900">${activity.activity_type_display}</p>
                        <p class="text-xs text-gray-500">${activity.change_summary}</p>
                        <p class="text-xs text-gray-400">${timestamp}</p>
                    </div>
                    <div class="flex-shrink-0 flex items-center">
                        <i class="fas fa-chevron-right text-gray-400 text-xs"></i>
                    </div>
                </div>
            `;
        });

        activityContainer.innerHTML = activitiesHtml;
        
        // Store activities data for modal access
        this.currentActivities = activities;
    }

    getActivityIcon(activityType) {
        const iconMap = {
            'created': 'fas fa-plus',
            'updated': 'fas fa-edit',
            'metrics_added': 'fas fa-chart-line',
            'metrics_removed': 'fas fa-chart-line',
            'parameters_changed': 'fas fa-cog',
            'name_changed': 'fas fa-edit',
            'description_changed': 'fas fa-align-left',
            'dataset_changed': 'fas fa-database',
            'job_ids_changed': 'fas fa-play',
            'result_ids_changed': 'fas fa-chart-bar',
            'sync_status_changed': 'fas fa-sync'
        };
        return iconMap[activityType] || 'fas fa-info-circle';
    }

    getActivityBgColor(activityType) {
        const colorMap = {
            'created': 'bg-green-100',
            'updated': 'bg-blue-100',
            'metrics_added': 'bg-green-100',
            'metrics_removed': 'bg-red-100',
            'parameters_changed': 'bg-blue-100',
            'name_changed': 'bg-blue-100',
            'description_changed': 'bg-blue-100',
            'dataset_changed': 'bg-purple-100',
            'job_ids_changed': 'bg-yellow-100',
            'result_ids_changed': 'bg-indigo-100',
            'sync_status_changed': 'bg-gray-100'
        };
        return colorMap[activityType] || 'bg-gray-100';
    }

    getActivityIconColor(activityType) {
        const colorMap = {
            'created': 'text-green-600',
            'updated': 'text-blue-600',
            'metrics_added': 'text-green-600',
            'metrics_removed': 'text-red-600',
            'parameters_changed': 'text-blue-600',
            'name_changed': 'text-blue-600',
            'description_changed': 'text-blue-600',
            'dataset_changed': 'text-purple-600',
            'job_ids_changed': 'text-yellow-600',
            'result_ids_changed': 'text-indigo-600',
            'sync_status_changed': 'text-gray-600'
        };
        return colorMap[activityType] || 'text-gray-600';
    }

    openActivityModal(activityId) {
        const activity = this.currentActivities?.find(a => a.id == activityId);
        if (!activity) return;

        const modal = document.getElementById('activity-modal');
        const modalTitle = document.getElementById('activity-modal-title');
        const modalIconContainer = document.getElementById('activity-modal-icon-container');
        const modalType = document.getElementById('activity-modal-type');
        const modalTimestamp = document.getElementById('activity-modal-timestamp');
        const modalSummary = document.getElementById('activity-modal-summary');
        const modalChanges = document.getElementById('activity-modal-changes');

        // Set up modal content
        modalTitle.textContent = `${activity.activity_type_display} Details`;
        modalType.textContent = activity.activity_type_display;
        modalTimestamp.textContent = new Date(activity.timestamp).toLocaleString();
        modalSummary.textContent = activity.change_summary;

        // Set up icon
        const icon = this.getActivityIcon(activity.activity_type);
        const bgColor = this.getActivityBgColor(activity.activity_type);
        const iconColor = this.getActivityIconColor(activity.activity_type);
        modalIconContainer.className = `flex-shrink-0 w-10 h-10 ${bgColor} rounded-full flex items-center justify-center`;
        modalIconContainer.innerHTML = `<i class="${icon} ${iconColor}"></i>`;

        // Generate changes details
        const changesHtml = this.generateChangesHtml(activity);
        modalChanges.innerHTML = changesHtml;

        // Show modal
        modal.classList.remove('hidden');

        // Add escape key listener
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                this.closeActivityModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    closeActivityModal() {
        const modal = document.getElementById('activity-modal');
        modal.classList.add('hidden');
    }

    generateChangesHtml(activity) {
        const changes = activity.changes || {};
        let changesHtml = '';

        switch (activity.activity_type) {
            case 'name_changed':
                changesHtml = this.generateValueChangeHtml('Name', changes.old_value, changes.new_value);
                break;
                
            case 'description_changed':
                changesHtml = this.generateValueChangeHtml('Description', changes.old_value, changes.new_value);
                break;
                
            case 'dataset_changed':
                changesHtml = this.generateValueChangeHtml('Dataset', changes.old_value, changes.new_value);
                break;
                
            case 'metrics_added':
                changesHtml = this.generateListChangeHtml('Added Metrics', changes.added || [], 'added');
                break;
                
            case 'metrics_removed':
                changesHtml = this.generateListChangeHtml('Removed Metrics', changes.removed || [], 'removed');
                break;
                
            case 'parameters_changed':
                changesHtml = this.generateJsonChangeHtml('Parameters', changes.old_value, changes.new_value);
                break;
                
            case 'created':
                changesHtml = `
                    <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                        <h5 class="font-medium text-green-800 mb-2">Method Created</h5>
                        <p class="text-sm text-green-700">This method was created and added to the system.</p>
                    </div>
                `;
                break;
                
            default:
                if (activity.description) {
                    changesHtml = `
                        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <h5 class="font-medium text-blue-800 mb-2">Details</h5>
                            <p class="text-sm text-blue-700">${activity.description}</p>
                        </div>
                    `;
                } else {
                    changesHtml = `
                        <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                            <p class="text-sm text-gray-600">No detailed change information available.</p>
                        </div>
                    `;
                }
                break;
        }

        return changesHtml;
    }

    generateValueChangeHtml(label, oldValue, newValue) {
        return `
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h5 class="font-medium text-blue-800 mb-3">${label} Change</h5>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-red-600 mb-1">Previous Value</label>
                        <div class="bg-red-50 border border-red-200 rounded p-3">
                            <code class="text-sm text-red-800">${this.escapeHtml(oldValue || 'Empty')}</code>
                        </div>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-green-600 mb-1">New Value</label>
                        <div class="bg-green-50 border border-green-200 rounded p-3">
                            <code class="text-sm text-green-800">${this.escapeHtml(newValue || 'Empty')}</code>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateListChangeHtml(label, items, changeType) {
        const colorClass = changeType === 'added' ? 'green' : 'red';
        const bgClass = `bg-${colorClass}-50`;
        const borderClass = `border-${colorClass}-200`;
        const textClass = `text-${colorClass}-800`;
        
        return `
            <div class="${bgClass} border ${borderClass} rounded-lg p-4">
                <h5 class="font-medium ${textClass} mb-3">${label}</h5>
                <div class="space-y-2">
                    ${items.map(item => `
                        <div class="bg-white border border-gray-200 rounded p-2">
                            <code class="text-sm text-gray-800">${this.escapeHtml(item)}</code>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    generateJsonChangeHtml(label, oldValue, newValue) {
        // Generate a diff-style comparison for JSON parameters
        const oldJson = oldValue ? JSON.stringify(oldValue, null, 2) : '{}';
        const newJson = newValue ? JSON.stringify(newValue, null, 2) : '{}';
        
        const diffHtml = this.generateJsonDiff(oldValue || {}, newValue || {});
        
        return `
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h5 class="font-medium text-blue-800 mb-3">${label} Change</h5>
                <div class="bg-white border border-gray-200 rounded p-3 max-h-96 overflow-y-auto">
                    <div class="space-y-1">
                        ${diffHtml}
                    </div>
                </div>
            </div>
        `;
    }

    generateJsonDiff(oldObj, newObj) {
        let diffHtml = '';
        const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);
        
        allKeys.forEach(key => {
            const oldValue = oldObj[key];
            const newValue = newObj[key];
            
            if (oldValue === undefined) {
                // Key was added
                diffHtml += `
                    <div class="flex items-start">
                        <span class="text-gray-500 font-mono text-sm mr-2 flex-shrink-0">+</span>
                        <div class="bg-green-50 border-l-4 border-green-400 pl-3 py-1 rounded-r flex-1 min-w-0">
                            <pre class="font-mono text-sm text-green-800 whitespace-pre-wrap break-words">"${this.escapeHtml(key)}": ${this.escapeHtml(JSON.stringify(newValue, null, 2))}</pre>
                        </div>
                    </div>
                `;
            } else if (newValue === undefined) {
                // Key was removed
                diffHtml += `
                    <div class="flex items-start">
                        <span class="text-gray-500 font-mono text-sm mr-2 flex-shrink-0">-</span>
                        <div class="bg-red-50 border-l-4 border-red-400 pl-3 py-1 rounded-r flex-1 min-w-0">
                            <pre class="font-mono text-sm text-red-800 whitespace-pre-wrap break-words">"${this.escapeHtml(key)}": ${this.escapeHtml(JSON.stringify(oldValue, null, 2))}</pre>
                        </div>
                    </div>
                `;
            } else if (JSON.stringify(oldValue) !== JSON.stringify(newValue)) {
                // Key was modified
                diffHtml += `
                    <div>
                        <div class="flex items-start">
                            <span class="text-gray-500 font-mono text-sm mr-2 flex-shrink-0">-</span>
                            <div class="bg-red-50 border-l-4 border-red-400 pl-3 py-1 rounded-r flex-1 min-w-0">
                                <pre class="font-mono text-sm text-red-800 whitespace-pre-wrap break-words">"${this.escapeHtml(key)}": ${this.escapeHtml(JSON.stringify(oldValue, null, 2))}</pre>
                            </div>
                        </div>
                        <div class="flex items-start">
                            <span class="text-gray-500 font-mono text-sm mr-2 flex-shrink-0">+</span>
                            <div class="bg-green-50 border-l-4 border-green-400 pl-3 py-1 rounded-r flex-1 min-w-0">
                                <pre class="font-mono text-sm text-green-800 whitespace-pre-wrap break-words">"${this.escapeHtml(key)}": ${this.escapeHtml(JSON.stringify(newValue, null, 2))}</pre>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Key unchanged
                diffHtml += `
                    <div class="flex items-start">
                        <span class="text-gray-400 font-mono text-sm mr-2 flex-shrink-0"> </span>
                        <div class="pl-3 py-1 flex-1 min-w-0">
                            <pre class="font-mono text-sm text-gray-600 whitespace-pre-wrap break-words">"${this.escapeHtml(key)}": ${this.escapeHtml(JSON.stringify(oldValue, null, 2))}</pre>
                        </div>
                    </div>
                `;
            }
        });
        
        return diffHtml;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    groupJobsByParameters(jobs) {
        const groups = [];
        const parameterGroups = new Map();

        jobs.forEach(job => {
            const paramKey = JSON.stringify(job.parameters || {});
            if (!parameterGroups.has(paramKey)) {
                parameterGroups.set(paramKey, {
                    parameters: job.parameters || {},
                    jobs: [],
                    created_at: null
                });
            }
            parameterGroups.get(paramKey).jobs.push(job);
        });
        
        // Sort jobs within each group in reverse chronological order (most recent first)
        // and set the group's created_at to the most recent job's timestamp
        parameterGroups.forEach(group => {
            // Sort jobs by created_at if available, otherwise by array position (assuming later = more recent)
            group.jobs.sort((a, b) => {
                const dateA = a.created_at ? new Date(a.created_at) : new Date(0);
                const dateB = b.created_at ? new Date(b.created_at) : new Date(0);
                return dateB - dateA; // Most recent first
            });
            
            // Set group's created_at to the most recent job's timestamp
            if (group.jobs.length > 0) {
                group.created_at = group.jobs[0].created_at || new Date();
            }
        });

        // Convert to array and sort groups by their created_at (most recent first)
        var list = Array.from(parameterGroups.values()).sort((a, b) => {
            const dateA = new Date(a.created_at);
            const dateB = new Date(b.created_at);
            return dateB - dateA; // Most recent group first
        });

        return list;
    }

    // Method to update the internal testsData when a job completes
    updateJobInData(jobData) {
        if (!jobData || !jobData.methodId) {
            console.warn('Invalid job data passed to updateJobInData:', jobData);
            return;
        }

        // Find the test that contains this method
        const test = this.testsData.find(test => 
            test.methods.some(method => method.id === jobData.methodId)
        );
        
        if (!test) {
            console.warn(`Test not found for method ${jobData.methodId}`);
            return;
        }

        // Find the specific method
        const method = test.methods.find(m => m.id === jobData.methodId);
        if (!method) {
            console.warn(`Method ${jobData.methodId} not found in test ${test.id}`);
            return;
        }

        // Initialize jobs array if it doesn't exist
        if (!method.jobs) {
            method.jobs = [];
        }

        // Check if this job already exists
        const existingJobIndex = method.jobs.findIndex(job => job.id === jobData.id);

        // Create the job object with the updated data
        const updatedJob = {
            id: jobData.id,
            status: jobData.status.toUpperCase(), // Convert to uppercase to match expected format
            status_message: jobData.status_message || '',
            type: jobData.type,
            parameters: jobData.parameters,
            result: jobData.result,
            result_id: jobData.result_id,
            log: jobData.log || [],
            runtime: jobData.runtime || 0,
            created_at: new Date().toISOString(),
        };

        if (existingJobIndex >= 0) {
            // Update existing job
            method.jobs[existingJobIndex] = updatedJob;
        } else {
            // Add new job
            method.jobs.push(updatedJob);
        }

        // Update result_ids if the job completed successfully and has results
        if (jobData.status === 'completed' && jobData.result_id) {
            if (!method.result_ids) {
                method.result_ids = [];
            }
            if (!method.result_ids.includes(jobData.result_id)) {
                method.result_ids.push(jobData.result_id);
            }
        }
    }

    getTestById(testId) {
        return this.testsData.find(test => test.id === testId);
    }

    switchToContent(htmlContent) {
        this.welcomeScreen.classList.add('hidden');
        this.dashboardContent.classList.remove('hidden');
        this.contentArea.innerHTML = htmlContent;
    }

    // Add visual feedback when content is refreshed due to job updates
    showUpdateFeedback() {
        if (this.contentArea) {
            this.contentArea.classList.add('update-flash');
            setTimeout(() => {
                this.contentArea.classList.remove('update-flash');
            }, 1000);
        }
    }

    showWelcome() {
        this.welcomeScreen.classList.remove('hidden');
        this.dashboardContent.classList.add('hidden');
        this.currentContent = null;
        
        // Clear all highlights when showing welcome screen
        this.clearAllHighlights();
    }

    renderTemplate(template, data) {
        let rendered = template;

        // Replace simple variables
        for (const [key, value] of Object.entries(data)) {
            if (Array.isArray(value)) {
                rendered = rendered.replace(new RegExp('\\{' + key + '\\}', 'g'), JSON.stringify(value));
            } else if (typeof value !== 'object') {
                rendered = rendered.replace(new RegExp('\\{' + key + '\\}', 'g'), value);
            }
        }
        
        // Handle methods array
        if (data.methods) {
            const methodsRegex = /\{#methods\}(.*?)\{\/methods\}/s;
            const methodsMatch = template.match(methodsRegex);
            
            if (methodsMatch) {
                let methodsHtml = data.methods.map(method => {
                    let methodTemplate = methodsMatch[1];
                    for (const [key, value] of Object.entries(method)) {
                        methodTemplate = methodTemplate.replace(new RegExp('\\{' + key + '\\}', 'g'), value);
                    }
                    return methodTemplate;
                }).join('');
                
                rendered = rendered.replace(methodsRegex, methodsHtml);
            }
        }

        // Handle cards array for method detail template
        if (data.cards) {
            const cardsRegex = /\{#cards\}(.*?)\{\/cards\}/s;
            const cardsMatch = template.match(cardsRegex);
            
            if (cardsMatch) {
                let cardsHtml = data.cards.map(card => {
                    let cardTemplate = cardsMatch[1];
                    for (const [key, value] of Object.entries(card)) {
                        cardTemplate = cardTemplate.replace(new RegExp('\\{' + key + '\\}', 'g'), value);
                    }
                    return cardTemplate;
                }).join('');
                
                rendered = rendered.replace(cardsRegex, cardsHtml);
            }
        }
        
        // Handle conditional blocks
        rendered = this.handleConditionals(rendered, data);
        
        return rendered;
    }

    handleConditionals(template, data) {
        let rendered = template;
        
        // Handle hasResults blocks
        const hasResultsRegex = /\{#hasResults\}(.*?)\{\/hasResults\}/gs;
        const noResultsRegex = /\{\^hasResults\}(.*?)\{\/hasResults\}/gs;
        
        rendered = rendered.replace(hasResultsRegex, (match, content) => {
            return data.hasResults ? content : '';
        });
        
        rendered = rendered.replace(noResultsRegex, (match, content) => {
            return !data.hasResults ? content : '';
        });
        
        return rendered;
    }
}

// Initialize with the tests data from Django
const contentManager = new ContentManager(testsData);

// Make contentManager globally available for the queue manager
window.contentManager = contentManager;

// Global functions for activity modal
window.openActivityModal = (activityId) => {
    contentManager.openActivityModal(activityId);
};

window.closeActivityModal = () => {
    contentManager.closeActivityModal();
};

// Global functions for job selection modal
let selectedJobId = null;
let pendingMethodIds = null;

window.showJobSelectionModal = (methodIds) => {
    pendingMethodIds = methodIds;
    selectedJobId = null;
    
    const modal = document.getElementById('job-selection-modal');
    const jobList = document.getElementById('job-selection-list');
    const noJobsMessage = document.getElementById('no-jobs-message');
    const runButton = document.getElementById('run-analysis-btn');
    
    // Get completed jobs for the methods
    const completedJobs = getCompletedJobsForMethods(JSON.parse(methodIds));
    
    if (completedJobs.length === 0) {
        jobList.classList.add('hidden');
        noJobsMessage.classList.remove('hidden');
        runButton.disabled = true;
    } else {
        jobList.classList.remove('hidden');
        noJobsMessage.classList.add('hidden');
        
        // Populate job list
        let jobListHtml = '';
        completedJobs.forEach(job => {
            const timestamp = contentManager.formatTimestamp(job.created_at);
            const runtime = contentManager.formatRuntime(job.runtime || 0);
            
            // Determine status styling
            let statusClass, statusIcon;
            if (job.status === 'COMPLETED' || job.status === 'SUCCESS') {
                statusClass = 'bg-green-100 text-green-800';
                statusIcon = 'fa-check-circle';
            } else if (job.status === 'FAILED') {
                statusClass = 'bg-red-100 text-red-800';
                statusIcon = 'fa-times-circle';
            } else if (job.status === 'CANCELED' || job.status === 'CANCELLED') {
                statusClass = 'bg-gray-100 text-gray-600';
                statusIcon = 'fa-ban';
            } else {
                statusClass = 'bg-yellow-100 text-yellow-800';
                statusIcon = 'fa-clock';
            }
            
            jobListHtml += `
                <div class="job-selection-item border border-gray-200 rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition-colors" 
                    data-job-id="${job.id}" onclick="selectJobForAnalysis('${job.id.replace(/'/g, "\\'")}')">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <div class="flex-shrink-0">
                                <label class="custom-radio" onclick="event.stopPropagation()">
                                    <input type="radio" name="job-selection" class="job-radio" data-job-id="${job.id}" onchange="selectJobForAnalysis('${job.id.replace(/'/g, "\\'")}')">
                                    <span class="radio-indicator"></span>
                                </label>
                            </div>
                            <div class="flex-1 min-w-0">
                                <div class="flex items-center space-x-2">
                                    <h4 class="font-medium text-gray-900">Job #${job.id}</h4>
                                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${statusClass}">
                                        <i class="fas ${statusIcon} mr-1"></i>
                                        ${job.status}
                                    </span>
                                </div>
                                <div class="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                                    <span><i class="fas fa-clock mr-1"></i>Runtime: ${runtime}</span>
                                    <span><i class="fas fa-calendar mr-1"></i>${timestamp}</span>
                                </div>
                                ${job.method_name ? `<div class="mt-1 text-sm text-gray-600">Method: ${job.method_name}</div>` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        jobList.innerHTML = jobListHtml;
        runButton.disabled = true;
    }
    
    modal.classList.remove('hidden');
    
    // Add escape key listener
    const escapeHandler = (e) => {
        if (e.key === 'Escape') {
            closeJobSelectionModal();
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
};

window.closeJobSelectionModal = () => {
    const modal = document.getElementById('job-selection-modal');
    modal.classList.add('hidden');
};

window.selectJobForAnalysis = (jobId) => {
    // Update selection state
    selectedJobId = jobId;
    
    // Update UI - check the radio button and update container styling
    document.querySelectorAll('.job-selection-item').forEach(item => {
        const radio = item.querySelector('.job-radio');
        if (item.getAttribute('data-job-id') == jobId) {
            item.classList.add('bg-purple-50', 'border-purple-300');
            radio.checked = true;
        } else {
            item.classList.remove('bg-purple-50', 'border-purple-300');
            radio.checked = false;
        }
    });
    
    // Enable run button
    document.getElementById('run-analysis-btn').disabled = false;
};

window.runSelectedAnalysis = () => {
    if (!selectedJobId || !pendingMethodIds) return;

    // Close the modal
    closeJobSelectionModal();
    
    // Run the analysis job with the selected job ID as a parameter
    runJob(pendingMethodIds, 'ANALYSIS', selectedJobId);
};

function getCompletedJobsForMethods(methodIds) {
    const completedJobs = [];
    
    // Search through the tests data to find completed jobs for the specified methods
    contentManager.testsData.forEach(test => {
        test.methods.forEach(method => {
            if (methodIds.includes(method.id) && method.jobs) {
                method.jobs.forEach(job => {
                    // Include both successful and failed jobs for analysis
                    if (job.status === 'COMPLETED' || job.status === 'SUCCESS' || job.status === 'FAILED' || job.status === 'CANCELED' || job.status === 'CANCELLED' || job.status === 'WARNING') {
                        completedJobs.push({
                            ...job,
                            method_name: method.name,
                            test_name: test.name
                        });
                    }
                });
            }
        });
    });
    
    // Sort by creation date (most recent first)
    completedJobs.sort((a, b) => {
        const dateA = new Date(a.created_at || 0);
        const dateB = new Date(b.created_at || 0);
        return dateB - dateA;
    });
    
    return completedJobs;
}

// Close activity modal when clicking outside of it
document.getElementById('activity-modal').addEventListener('click', (e) => {
    if (e.target.id === 'activity-modal') {
        closeActivityModal();
    }
});

// Close job details modal when clicking outside of it
document.getElementById('job-details-modal').addEventListener('click', (e) => {
    if (e.target.id === 'job-details-modal') {
        closeJobDetailsModal();
    }
});

// Close job selection modal when clicking outside of it
document.getElementById('job-selection-modal').addEventListener('click', (e) => {
    if (e.target.id === 'job-selection-modal') {
        closeJobSelectionModal();
    }
});

// Sync checkbox states between modal and main view
document.addEventListener('change', (e) => {
    if (e.target.classList.contains('job-checkbox')) {
        const jobId = e.target.getAttribute('data-job-id');
        const allCheckboxes = document.querySelectorAll(`.job-checkbox[data-job-id="${jobId}"]`);
        
        allCheckboxes.forEach(checkbox => {
            if (checkbox !== e.target) {
                checkbox.checked = e.target.checked;
            }
        });
        
        updateCompareButton();
    }
});

// Updated functions for sidebar clicks
async function showDashboardContent(itemName, type = 'test', itemId = null, testId = null) {
    if (type === 'test') {
        if (contentManager.currentContent?.type === 'test' && contentManager.currentContent?.id === itemId) {
            contentManager.showWelcome();
        } else {
            contentManager.showTestContent(itemId);
        }
    } else if (type === 'method') {
        if (contentManager.currentContent?.type === 'method' && contentManager.currentContent?.id === itemId) {
            contentManager.showWelcome();
        } else {
            await contentManager.showMethodContent(itemId, testId);
        }
    }
}

async function runJob(methods, jobType = 'FULL', analysisJobId = null) {
    // Get the run button that was clicked
    const runButton = document.getElementById('run-button');
    const runButtonOptions = document.getElementById('run-button-options');

    if (!runButton) return;
    
    const originalContent = runButton.innerHTML;
    const originalClasses = runButton.className;
    if (runButtonOptions) var originalClassesOptions = runButtonOptions.className;

    // Show success feedback with animation
    runButton.innerHTML = '<i class="fas fa-check mr-2"></i>Queued';
    runButton.className = runButton.className.replace(/bg-primary|hover:bg-primary\/90/, 'bg-green-500 hover:bg-green-600 run-button-success');
    runButton.disabled = true;

    if (runButtonOptions) {
        runButtonOptions.className = runButtonOptions.className.replace(/bg-primary|hover:bg-primary\/90/, 'bg-green-500 hover:bg-green-600 run-button-success');
        runButtonOptions.disabled = true;
    }

    try {
        const response = await sendJobRequest(methods, jobType, analysisJobId);
        const result = await response.json();
        
        // Show completion state briefly
        runButton.innerHTML = '<i class="fas fa-check-circle mr-2"></i>Submitted';
        
        // Add jobs to queue using the API response data
        if (window.queueManager && result.jobs) {
            // Get method and test information for better job display names
            let jobData = { jobs: result.jobs };
            
            if (contentManager.currentContent) {
                const currentData = contentManager.currentContent;
                
                if (currentData.type === 'method') {
                    const testData = contentManager.getTestById(currentData.testId);
                    const methodData = testData?.methods.find(m => m.id === currentData.id);
                    jobData.methodName = methodData?.name || 'Unknown Method';
                    jobData.testName = testData?.name || 'Unknown Test';
                } else if (currentData.type === 'test') {
                    const testData = contentManager.getTestById(currentData.id);
                    jobData.methodName = `${result.jobs.length} method${result.jobs.length > 1 ? 's' : ''}`;
                    jobData.testName = testData?.name || 'Unknown Test';
                }
            }
            window.queueManager.addJob(jobData);
        }
        
        // Add the new jobs to contentManager.testsData so they appear in the dashboard immediately
        if (window.contentManager && result.jobs) {
            result.jobs.forEach(job => {
                // Convert snake_case API response to camelCase
                const jobData = {
                    id: job.id,
                    methodId: job.method_id,
                    status: job.status,
                    status_message: job.status_message,
                    type: job.type,
                    parameters: job.parameters,
                    result: job.result,
                    result_id: job.result_id,
                    log: job.log,
                    runtime: job.runtime,
                    created_at: job.created_at
                };
                window.contentManager.updateJobInData(jobData);
            });
        }
        
        // Refresh the dashboard content to show the new job
        if (window.contentManager && window.contentManager.currentContent) {
            const currentContent = window.contentManager.currentContent;
            
            // If we're viewing a method, refresh the method content
            if (currentContent.type === 'method') {
                window.contentManager.showMethodContent(currentContent.id, currentContent.testId);
            }
            // If we're viewing a test, refresh the test content
            else if (currentContent.type === 'test') {
                window.contentManager.showTestContent(currentContent.id);
            }
            
            // Show visual feedback that content was refreshed
            if (typeof window.contentManager.showUpdateFeedback === 'function') {
                window.contentManager.showUpdateFeedback();
            }
        }
        
        // Reset button after a short delay
        setTimeout(() => {
            runButton.innerHTML = originalContent;
            runButton.className = originalClasses;
            runButton.disabled = false;
            if (runButtonOptions) runButtonOptions.className = originalClassesOptions;
            if (runButtonOptions) runButtonOptions.disabled = false;
        }, 2000);
        
    } catch (error) {
        console.error('Error running job:', error);
        
        // Show error state
        runButton.innerHTML = '<i class="fas fa-times mr-2"></i>Error';
        runButton.className = runButton.className.replace(/bg-green-500|hover:bg-green-600|run-button-success/, 'bg-red-500 hover:bg-red-600 run-button-error');
        
        // Reset button after error
        setTimeout(() => {
            runButton.innerHTML = originalContent;
            runButton.className = originalClasses;
            runButton.disabled = false;
        }, 3000);
    }
}

function sendJobRequest(methodIds, jobType = 'FULL', analysisJobId = null) {
    if (jobType === 'ANALYSIS' ) {
        const requestBody = {
            job_ids: [analysisJobId],
            job_types: [jobType]
        };

        return fetch(duplicateJobsApi, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'), // Django CSRF protection
            },
            body: JSON.stringify(requestBody)
        });
    }

    const requestBody = {
        method_ids: JSON.parse(methodIds),
        job_types: [jobType]
    };

    return fetch(createJobsApi, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'), // Django CSRF protection
        },
        body: JSON.stringify(requestBody)
    });
}

// Helper function for CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function editMethod(projectId, methodId) {
    window.location.href = `/projects/${projectId}/methods/${methodId}/edit/`;
}

function editTest(projectId, testId) {
    window.location.href = `/projects/${projectId}/tests/${testId}/edit/`;
}

// Dropdown functionality for Run button
function toggleRunDropdown(event) {
    event.stopPropagation();
    const dropdown = event.target.closest('.relative').querySelector('.run-dropdown');
    
    // Close other dropdowns first
    document.querySelectorAll('.run-dropdown').forEach(dd => {
        if (dd !== dropdown) {
            dd.classList.add('hidden');
        }
    });
    
    // Toggle current dropdown
    dropdown.classList.toggle('hidden');
    
    // Focus first option when opening
    if (!dropdown.classList.contains('hidden')) {
        const firstOption = dropdown.querySelector('button');
        if (firstOption) {
            firstOption.focus();
        }
    }
}

function selectRunOption(methodIds, event, jobType) {
    // Close the dropdown
    const dropdown = event.target.closest('.run-dropdown');
    dropdown.classList.add('hidden');
    
    showJobSelectionModal(methodIds);
}

// Dropdown functionality for Export button
function toggleExportDropdown(event) {
    event.stopPropagation();
    const dropdown = event.target.closest('.relative').querySelector('.export-dropdown');
    
    // Close other dropdowns first
    document.querySelectorAll('.export-dropdown').forEach(dd => {
        if (dd !== dropdown) {
            dd.classList.add('hidden');
        }
    });
    
    // Toggle current dropdown
    dropdown.classList.toggle('hidden');
    
    // Focus first option when opening
    if (!dropdown.classList.contains('hidden')) {
        const firstOption = dropdown.querySelector('button');
        if (firstOption) {
            firstOption.focus();
        }
    }
}

// Dropdown functionality for Diagnostic button
function toggleDiagnosticDropdown(event) {
    event.stopPropagation();
    const dropdown = event.target.closest('.relative').querySelector('.diagnostic-dropdown');
    
    // Close other dropdowns first
    document.querySelectorAll('.diagnostic-dropdown, .run-dropdown, .export-dropdown').forEach(dd => {
        if (dd !== dropdown) {
            dd.classList.add('hidden');
        }
    });
    
    // Toggle current dropdown
    dropdown.classList.toggle('hidden');
    
    // Focus first option when opening
    if (!dropdown.classList.contains('hidden')) {
        const firstOption = dropdown.querySelector('button');
        if (firstOption) {
            firstOption.focus();
        }
    }
}

// Close dropdowns when clicking outside or pressing escape
document.addEventListener('click', function() {
    document.querySelectorAll('.run-dropdown, .export-dropdown, .diagnostic-dropdown').forEach(dropdown => {
        dropdown.classList.add('hidden');
    });
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        document.querySelectorAll('.run-dropdown, .export-dropdown, .diagnostic-dropdown').forEach(dropdown => {
            dropdown.classList.add('hidden');
        });
    }
});

// Document modal functions
function showDocumentModal(text, docId) {
    document.getElementById('modalDocId').textContent = docId;
    document.getElementById('modalDocText').textContent = text;
    document.getElementById('documentModal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeDocumentModal() {
    document.getElementById('documentModal').classList.add('hidden');
    document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
document.addEventListener('click', function(event) {
    const modal = document.getElementById('documentModal');
    if (event.target === modal) {
        closeDocumentModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeDocumentModal();
    }
});

// Toggle functions for collapsible output and reference sections
let inputFullText = '';
let outputFullText = '';
let referenceFullText = '';
let inputExpanded = false;
let outputExpanded = false;
let referenceExpanded = false;

function truncateToWords(text, wordCount) {
    if (!text) return '';
    const words = text.split(' ');
    if (words.length <= wordCount) return text;
    return words.slice(0, wordCount).join(' ') + '...';
}

function setInputContent(fullText) {
    inputFullText = fullText;
    const inputContent = document.getElementById('job-input-content');
    const inputToggleBtn = document.getElementById('input-toggle-btn');
    
    // Handle non-string values
    if (!fullText || typeof fullText !== 'string') {
        if (inputContent) inputContent.innerHTML = '<span class="text-gray-400">No input available</span>';
        if (inputToggleBtn) inputToggleBtn.classList.add('hidden');
        return;
    }

    const words = fullText.split(' ');
    if (words.length <= 100) {
        // Short enough to show all
        if (inputContent) inputContent.innerHTML = `<div class="text-sm">${fullText}</div>`;
        if (inputToggleBtn) inputToggleBtn.classList.add('hidden');
    } else {
        // Show truncated version with toggle button
        if (inputContent) inputContent.innerHTML = `<div class="text-sm">${truncateToWords(fullText, 100)}</div>`;
        if (inputToggleBtn) {
            inputToggleBtn.classList.remove('hidden');
            inputToggleBtn.innerHTML = '<i class="fas fa-chevron-down mr-1"></i>Show More';
        }
    }
    inputExpanded = false;
}

function setOutputContent(fullText) {
    outputFullText = fullText;
    const outputContent = document.getElementById('job-output-content');
    const outputToggleBtn = document.getElementById('output-toggle-btn');
    
    // Handle non-string values
    if (!fullText || typeof fullText !== 'string') {
        outputContent.innerHTML = '<span class="text-gray-400">No output available</span>';
        outputToggleBtn.classList.add('hidden');
        return;
    }

    const words = fullText.split(' ');
    if (words.length <= 100) {
        // Short text, no need for toggle
        outputContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(fullText)}</div>`;
        outputToggleBtn.classList.add('hidden');
    } else {
        // Long text, show truncated version
        const truncated = truncateToWords(fullText, 100);
        outputContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(truncated)}</div>`;
        outputToggleBtn.classList.remove('hidden');
        outputToggleBtn.textContent = 'Show more';
    }
    outputExpanded = false;
}

function setReferenceContent(fullText) {
    referenceFullText = fullText;
    const referenceContent = document.getElementById('job-reference-content');
    const referenceToggleBtn = document.getElementById('reference-toggle-btn');
    
    // Handle non-string values
    if (!fullText || typeof fullText !== 'string') {
        referenceContent.innerHTML = '<span class="text-gray-400">No reference available</span>';
        referenceToggleBtn.classList.add('hidden');
        return;
    }

    const words = fullText.split(' ');
    if (words.length <= 100) {
        // Short text, no need for toggle
        referenceContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(fullText)}</div>`;
        referenceToggleBtn.classList.add('hidden');
    } else {
        // Long text, show truncated version
        const truncated = truncateToWords(fullText, 100);
        referenceContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(truncated)}</div>`;
        referenceToggleBtn.classList.remove('hidden');
        referenceToggleBtn.textContent = 'Show more';
    }
    referenceExpanded = false;
}

function toggleInputExpansion() {
    const inputContent = document.getElementById('job-input-content');
    const inputToggleBtn = document.getElementById('input-toggle-btn');
    
    if (inputExpanded) {
        // Collapse
        inputContent.innerHTML = `<div class="text-sm">${truncateToWords(inputFullText, 100)}</div>`;
        inputToggleBtn.innerHTML = '<i class="fas fa-chevron-down mr-1"></i>Show More';
        inputExpanded = false;
    } else {
        // Expand
        inputContent.innerHTML = `<div class="text-sm">${inputFullText}</div>`;
        inputToggleBtn.innerHTML = '<i class="fas fa-chevron-up mr-1"></i>Show Less';
        inputExpanded = true;
    }
}

function toggleOutputExpansion() {
    const outputContent = document.getElementById('job-output-content');
    const outputToggleBtn = document.getElementById('output-toggle-btn');
    
    if (outputExpanded) {
        // Show truncated version
        const truncated = truncateToWords(outputFullText, 100);
        outputContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(truncated)}</div>`;
        outputToggleBtn.textContent = 'Show more';
        outputExpanded = false;
    } else {
        // Show full text
        outputContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(outputFullText)}</div>`;
        outputToggleBtn.textContent = 'Show less';
        outputExpanded = true;
    }
}

function toggleReferenceExpansion() {
    const referenceContent = document.getElementById('job-reference-content');
    const referenceToggleBtn = document.getElementById('reference-toggle-btn');
    
    if (referenceExpanded) {
        // Show truncated version
        const truncated = truncateToWords(referenceFullText, 100);
        referenceContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(truncated)}</div>`;
        referenceToggleBtn.textContent = 'Show more';
        referenceExpanded = false;
    } else {
        // Show full text
        referenceContent.innerHTML = `<div class="text-sm">${contentManager.escapeHtml(referenceFullText)}</div>`;
        referenceToggleBtn.textContent = 'Show less';
        referenceExpanded = true;
    }
}

// Diagnostic Functions
let currentJobForDiagnostic = null;
let currentSampleForDiagnostic = null;

// Global function for running diagnostics
async function runDiagnostic(diagnosticType, scope) {
    if (!currentJobForDiagnostic) {
        alert('No job selected for diagnostic');
        return;
    }

    // If it's a sample diagnostic but no sample is available, show alert
    if (scope === 'sample' && !currentSampleForDiagnostic) {
        alert('No sample selected. Please navigate to a specific sample first.');
        return;
    }

    // Close the dropdown if it's open
    const diagnosticDropdown = document.querySelector('.diagnostic-dropdown');
    if (diagnosticDropdown) {
        diagnosticDropdown.classList.add('hidden');
    }

    const diagnosticButton = document.getElementById('run-diagnostic-btn');
    const diagnosticOptions = document.getElementById('run-diagnostic-options');
    
    // Disable buttons during execution
    const originalContent = diagnosticButton.innerHTML;
    const originalClasses = diagnosticButton.className;
    const originalOptionsClasses = diagnosticOptions.className;
    
    diagnosticButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Running...';
    diagnosticButton.disabled = true;
    diagnosticOptions.disabled = true;

    try {
        const requestBody = {
            job_id: currentJobForDiagnostic.id,
            diagnostic: diagnosticType
        };

        // If running on sample, add the current sample data
        if (scope === 'sample' && currentSampleForDiagnostic) {
            requestBody.sample = currentSampleForDiagnostic;
        }

        const response = await fetch(createJobDiagnosticApi, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        const result = await response.json();

        if (response.ok) {
            // Show success feedback
            diagnosticButton.innerHTML = '<i class="fas fa-check mr-2"></i>Queued';
            diagnosticButton.className = diagnosticButton.className.replace(/bg-purple-600|hover:bg-purple-700/, 'bg-green-500 hover:bg-green-600');
            diagnosticOptions.className = diagnosticOptions.className.replace(/bg-purple-600|hover:bg-purple-700/, 'bg-green-500 hover:bg-green-600');
            
            // Add job to queue manager
            if (window.queueManager && result.job) {
                const diagnosticName = diagnosticType.toUpperCase();
                const scopeName = scope === 'full' ? 'Full Job' : 'Sample';
                const jobData = {
                    jobs: [result.job],
                    methodName: `${diagnosticName} Diagnostic (${scopeName})`,
                    testName: currentJobForDiagnostic.methodName || 'Diagnostic'
                };
                window.queueManager.addJob(jobData);
            }

            // Reset button after delay
            setTimeout(() => {
                diagnosticButton.innerHTML = originalContent;
                diagnosticButton.className = originalClasses;
                diagnosticOptions.className = originalOptionsClasses;
                diagnosticButton.disabled = false;
                diagnosticOptions.disabled = false;
            }, 2000);

        } else {
            throw new Error(result.error || `Failed to create ${diagnosticType.toUpperCase()} diagnostic job`);
        }

    } catch (error) {
        console.error(`Error running ${diagnosticType.toUpperCase()} diagnostic:`, error);
        alert(`Error running ${diagnosticType.toUpperCase()} diagnostic: ` + error.message);
        
        // Reset buttons on error
        diagnosticButton.innerHTML = originalContent;
        diagnosticButton.className = originalClasses;
        diagnosticOptions.className = originalOptionsClasses;
        diagnosticButton.disabled = false;
        diagnosticOptions.disabled = false;
    }
}

// Helper function for CSRF token (if not already defined)
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Global function for toggling diagnosis report expansion
function toggleDiagnosisExpansion(index) {
    const reportElement = document.getElementById(`diagnosis-report-${index}`);
    const toggleButton = document.getElementById(`diagnosis-toggle-${index}`);
    
    if (!contentManager.currentDiagnosticReports || !contentManager.currentDiagnosticReports[index]) {
        console.error('Diagnostic report data not found for index:', index);
        return;
    }
    
    const reportData = contentManager.currentDiagnosticReports[index];
    
    // Only handle text-based reports, LXT reports don't need expansion
    if (reportData.type !== 'text') {
        return;
    }
    
    const isExpanded = toggleButton.innerHTML.includes('Show Less');
    
    if (isExpanded) {
        reportElement.textContent = reportData.preview;
        toggleButton.innerHTML = '<i class="fas fa-chevron-down mr-1"></i>Show Full Report';
    } else {
        reportElement.textContent = reportData.full;
        toggleButton.innerHTML = '<i class="fas fa-chevron-up mr-1"></i>Show Less';
    }
}

// Global functions for tag management
async function addTagToJob() {
    const select = document.getElementById('available-tags-select');
    const selectedTagName = select.value;
    
    if (!selectedTagName || !contentManager.currentTagJob) {
        console.error('No tag selected or no job available');
        return;
    }
    
    const addBtn = document.getElementById('add-tag-btn');
    const originalContent = addBtn.innerHTML;
    addBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Adding...';
    addBtn.disabled = true;
    
    try {
        const response = await fetch('/api/tags/create/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                tag_name: selectedTagName,
                job_id: contentManager.currentTagJob.id,
                comment: '',
                filters: ['SK', 'RK']
            })
        });
        
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const error = await response.json();
                errorMessage = error.error || errorMessage;
            } catch (e) {
                // Response is not JSON (probably HTML error page)
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        // Reset form
        select.value = '';
        contentManager.hideTagPreview();
        
        // Reload current tags
        await contentManager.loadCurrentTags(contentManager.currentTagJob.id);
        
        // Update search data if the function is available
        if (window.updateSearchDataForMethod && contentManager.currentTagMethodId) {
            window.updateSearchDataForMethod(contentManager.currentTagMethodId);
        } else {
            console.warn('Cannot update search data:', {
                functionExists: !!window.updateSearchDataForMethod,
                methodId: contentManager.currentTagMethodId
            });
        }
        
        // Show success feedback
        addBtn.innerHTML = '<i class="fas fa-check mr-2"></i>Added';
        addBtn.className = addBtn.className.replace(/bg-primary/, 'bg-green-500');
        
        setTimeout(() => {
            addBtn.innerHTML = originalContent;
            addBtn.className = addBtn.className.replace(/bg-green-500/, 'bg-primary');
            addBtn.disabled = true;
        }, 2000);
        
    } catch (error) {
        console.error('Error adding tag:', error);
        alert('Error adding tag: ' + error.message);
        
        addBtn.innerHTML = originalContent;
        addBtn.disabled = false;
    }
}

async function removeTagFromJob(tagId) {
    if (!confirm('Are you sure you want to remove this tag?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/tags/delete/${tagId}/`, {
            method: 'DELETE',
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        });
        
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const error = await response.json();
                errorMessage = error.error || errorMessage;
            } catch (e) {
                // Response is not JSON (probably HTML error page)
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        // Reload current tags
        await contentManager.loadCurrentTags(contentManager.currentTagJob.id);
        
        // Update search data if the function is available
        if (window.updateSearchDataForMethod && contentManager.currentTagMethodId) {
            window.updateSearchDataForMethod(contentManager.currentTagMethodId);
        }
        
    } catch (error) {
        console.error('Error removing tag:', error);
        alert('Error removing tag: ' + error.message);
    }
}

async function updateTagComment(tagId, comment) {
    try {
        const response = await fetch(`/api/tags/update/${tagId}/`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                comment: comment
            })
        });
        
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const error = await response.json();
                errorMessage = error.error || errorMessage;
            } catch (e) {
                // Response is not JSON (probably HTML error page)
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        // Show success notification
        showCommentNotification(tagId, 'success');
        
    } catch (error) {
        console.error('Error updating tag comment:', error);
        // Show error notification
        showCommentNotification(tagId, 'error');
    }
}

function showCommentNotification(tagId, type) {
    const textarea = document.getElementById(`tag-comment-${tagId}`);
    const statusSpan = document.getElementById(`comment-status-${tagId}`);
    
    if (!textarea || !statusSpan) return;
    
    if (type === 'success') {
        // Show success status
        statusSpan.textContent = 'Comment saved';
        statusSpan.className = 'text-xs font-medium text-green-600 transition-all duration-300';
        statusSpan.classList.remove('hidden');
        
        // Add subtle green border to textarea
        textarea.style.borderColor = '#10b981';
        textarea.style.transition = 'border-color 0.3s ease';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            statusSpan.classList.add('hidden');
            textarea.style.borderColor = '';
        }, 3000);
        
    } else if (type === 'error') {
        // Show error status
        statusSpan.textContent = 'Failed to save';
        statusSpan.className = 'text-xs font-medium text-red-600 transition-all duration-300';
        statusSpan.classList.remove('hidden');
        
        // Add subtle red border to textarea
        textarea.style.borderColor = '#ef4444';
        textarea.style.transition = 'border-color 0.3s ease';
        
        // Auto-hide after 4 seconds (longer for errors)
        setTimeout(() => {
            statusSpan.classList.add('hidden');
            textarea.style.borderColor = '';
        }, 4000);
    }
}