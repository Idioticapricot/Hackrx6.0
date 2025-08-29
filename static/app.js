// LegalEase AI - Frontend JavaScript

let currentDocument = null;
let currentAnalysis = null;

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // File upload handling
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const urlInput = document.getElementById('urlInput');

    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // URL input handling
    urlInput.addEventListener('input', updateFileInfo);
}

// File handling
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = e.dataTransfer.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    handleFiles(e.target.files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        document.getElementById('fileInfo').innerHTML = 
            `<div class="file-info">📎 Selected: ${file.name}</div>`;
        document.getElementById('urlInput').value = '';
        currentDocument = file;
    }
}

function updateFileInfo() {
    const url = document.getElementById('urlInput').value;
    if (url) {
        document.getElementById('fileInfo').innerHTML = 
            `<div class="file-info">🔗 URL: ${url}</div>`;
        document.getElementById('fileInput').value = '';
        currentDocument = url;
    } else {
        document.getElementById('fileInfo').innerHTML = '';
        currentDocument = null;
    }
}

// Document analysis
async function analyzeDocument() {
    const url = document.getElementById('urlInput').value.trim();
    const language = document.getElementById('languageSelect').value;
    
    if (!url && !currentDocument) {
        alert('Please provide a document URL or upload a file');
        return;
    }
    
    if (!url) {
        alert('File upload requires server implementation. Please use URL input for now.');
        return;
    }
    
    showLoading('Analyzing legal document with AI...');
    
    try {
        // Perform legal analysis
        const response = await fetch('/legal/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                document_url: url,
                language: language
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentAnalysis = data.analysis;
        
        displayAnalysis(currentAnalysis);
        showMainContent();
        
    } catch (error) {
        hideLoading();
        alert(`Error: ${error.message}`);
    }
}

function displayAnalysis(analysis) {
    hideLoading();
    
    // Display simplified content
    displaySimplifiedContent(analysis.simplified_text, analysis.risks);
    
    // Display risks
    displayRisks(analysis.risks, analysis.risk_summary);
    
    // Show the main content area
    document.getElementById('mainContent').style.display = 'block';
    showTab('simplified');
}

function displaySimplifiedContent(simplifiedText, risks) {
    const container = document.getElementById('simplifiedContent');
    const riskSummary = document.getElementById('riskSummary');
    
    // Display risk summary
    if (risks && risks.length > 0) {
        const overallRisk = determineOverallRisk(risks);
        riskSummary.className = `risk-summary ${overallRisk}`;
        riskSummary.innerHTML = `
            <div class="risk-level">${getRiskIcon(overallRisk)} ${overallRisk.toUpperCase()} RISK DOCUMENT</div>
            <p>Found ${risks.length} potential issues. Review the Risk Analysis tab for details.</p>
        `;
    } else {
        riskSummary.className = 'risk-summary low';
        riskSummary.innerHTML = `
            <div class="risk-level">✅ LOW RISK DOCUMENT</div>
            <p>No significant risks detected in this document.</p>
        `;
    }
    
    // Display the complete simplified/rewritten document
    container.innerHTML = `
        <div class="simplified-document">
            <h3>📄 Simplified Document</h3>
            <div class="simplified-text">${formatSimplifiedText(simplifiedText)}</div>
        </div>
    `;
}

function displayRisks(risks, riskSummary) {
    const container = document.getElementById('risksContainer');
    
    if (!risks || risks.length === 0) {
        container.innerHTML = `
            <div class="risk-item low">
                <div class="risk-header">
                    <span class="risk-icon">✅</span>
                    <span class="risk-title">No Significant Risks Detected</span>
                </div>
                <div class="risk-description">
                    This document appears to contain standard terms and conditions without major red flags.
                </div>
            </div>
        `;
        return;
    }
    
    let html = '';
    risks.forEach(risk => {
        const icon = getRiskIcon(risk.level);
        html += `
            <div class="risk-item ${risk.level}">
                <div class="risk-header">
                    <span class="risk-icon">${icon}</span>
                    <span class="risk-title">${risk.level.toUpperCase()} RISK</span>
                </div>
                <div class="risk-description">${risk.description}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function formatSimplifiedText(text) {
    if (!text) return '<p>No simplified version available.</p>';
    
    // Format the simplified text with proper paragraphs and structure
    const paragraphs = text.split('\n\n').filter(p => p.trim());
    
    let formattedText = '';
    paragraphs.forEach(paragraph => {
        const trimmed = paragraph.trim();
        if (trimmed) {
            // Check if it's a heading (starts with numbers, bullets, or all caps)
            if (trimmed.match(/^\d+\.|^[A-Z\s]{10,}$|^\*|^-/)) {
                formattedText += `<h4 class="section-heading">${trimmed}</h4>`;
            } else {
                formattedText += `<p class="simplified-paragraph">${trimmed}</p>`;
            }
        }
    });
    
    return formattedText || '<p>Document simplification in progress...</p>';
}

function determineOverallRisk(risks) {
    if (!risks || risks.length === 0) return 'low';
    
    const hasHigh = risks.some(r => r.level === 'high');
    const hasMedium = risks.some(r => r.level === 'medium');
    
    if (hasHigh) return 'high';
    if (hasMedium) return 'medium';
    return 'low';
}

function getRiskIcon(level) {
    switch (level) {
        case 'high': return '🔴';
        case 'medium': return '🟡';
        case 'low': return '🟢';
        default: return '⚪';
    }
}

// Tab management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + 'Tab').style.display = 'block';
    
    // Add active class to corresponding button
    const buttons = document.querySelectorAll('.tab-btn');
    const tabIndex = ['simplified', 'risks', 'qa'].indexOf(tabName);
    if (tabIndex !== -1 && buttons[tabIndex]) {
        buttons[tabIndex].classList.add('active');
    }
}

// Q&A functionality
async function askQuestion(question) {
    if (!currentAnalysis) {
        alert('Please analyze a document first');
        return;
    }
    
    showLoading('Getting answer...');
    
    try {
        const language = document.getElementById('languageSelect').value;
        const url = document.getElementById('urlInput').value;
        
        const response = await fetch('/hackathon', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                documents: url,
                questions: [question]
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayQAResult(question, data.answers[0]);
        
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function askCustomQuestion() {
    const question = document.getElementById('customQuestion').value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }
    
    askQuestion(question);
    document.getElementById('customQuestion').value = '';
}

function handleQuestionEnter(event) {
    if (event.key === 'Enter') {
        askCustomQuestion();
    }
}

function displayQAResult(question, answer) {
    const container = document.getElementById('qaResults');
    
    const qaItem = document.createElement('div');
    qaItem.className = 'qa-item';
    qaItem.innerHTML = `
        <div class="question">Q: ${question}</div>
        <div class="answer">A: ${answer}</div>
    `;
    
    container.insertBefore(qaItem, container.firstChild);
}

// UI helpers
function showLoading(message) {
    document.getElementById('loadingText').textContent = message;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showMainContent() {
    document.getElementById('mainContent').style.display = 'block';
    document.querySelector('.upload-section').style.display = 'none';
}

// Language change handler
document.getElementById('languageSelect').addEventListener('change', function() {
    if (currentAnalysis) {
        // Re-analyze with new language
        analyzeDocument();
    }
});