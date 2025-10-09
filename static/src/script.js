// script.js
// Decide the API endpoint based on local dev or production
const isLocal = window.location.hostname === "localhost";
const API_PREFIX = isLocal
  ? "http://0.0.0.0:8000"  // local dev
//   : "https://my-medical-chatbot.onrender.com";     // production Render server
//   : "https://medical-chatbot-henna.streamlit.app"; // production Streamlit server
     : "https://BinKhoaLe1812-Medical-Chatbot.hf.space"


// Test markdown rendering
console.log("Testing markdown rendering...");
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false
});
console.log("Simple heading:", marked.parse("### Simple Heading"));
console.log("Bold text:", marked.parse("**Bold Text**"));
console.log("Nested formatting:", marked.parse("### **Bold Heading**"));
console.log("Mixed formatting:", marked.parse("### **Bold** and *italic* heading"));

// Global variable for current language (default English)
let currentLang = "EN";

// Global variable for current theme (default light)
let currentTheme = "light";

// Global variables for independent mode states
let searchModeActive = false;
let uploadModeActive = false;
let videoModeActive = false;

// Submission state management
let isSubmitting = false;
let lastSubmissionTime = 0;
const SUBMISSION_DEBOUNCE_MS = 1000; // Prevent rapid successive submissions

// Chat history management
const CHAT_HISTORY_KEY = 'medical_chatbot_history';
const PENDING_REQUESTS_KEY = 'medical_chatbot_pending_requests';
const MAX_HISTORY_ITEMS = 100; // Limit to prevent localStorage bloat
const MAX_PENDING_REQUESTS = 5; // Limit pending requests

// Chat history functions
function saveChatHistory() {
    try {
        const messagesDiv = document.getElementById('chat-messages');
        const messages = Array.from(messagesDiv.children)
            .filter(messageEl => messageEl.classList.contains('message'))
            .map(messageEl => {
                // Determine role based on message structure
                const userElement = messageEl.querySelector('.user');
                const botElement = messageEl.querySelector('.bot');
                const role = userElement ? 'user' : (botElement ? 'bot' : 'unknown');
                
                // Get the message content (excluding the role label)
                const messageBubble = messageEl.querySelector('.message-bubble');
                const content = messageBubble ? messageBubble.innerHTML : messageEl.innerHTML;
                
                const timestamp = new Date().toISOString();
                
                return {
                    role,
                    content,
                    timestamp
                };
            })
            .filter(msg => msg.role !== 'unknown'); // Filter out unknown role messages
        
        // Keep only the last MAX_HISTORY_ITEMS messages
        const limitedMessages = messages.slice(-MAX_HISTORY_ITEMS);
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(limitedMessages));
        
        console.log(`Saved ${limitedMessages.length} messages to chat history`);
    } catch (error) {
        console.error('Error saving chat history:', error);
    }
}

function loadChatHistory() {
    try {
        const savedHistory = localStorage.getItem(CHAT_HISTORY_KEY);
        if (savedHistory) {
            const messages = JSON.parse(savedHistory);
            const messagesDiv = document.getElementById('chat-messages');
            
            // Clear existing messages (except welcome screen)
            const welcomeContainer = document.getElementById('welcome-container');
            if (welcomeContainer) {
                welcomeContainer.remove();
            }
            
            // Load each message with proper structure
            messages.forEach(messageData => {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('message');
                
                // Create message structure similar to appendMessage
                const prefix = messageData.role === 'user' ? translations[currentLang].you : translations[currentLang].bot;
                
                // Create message container
                const messageContainer = document.createElement('div');
                messageContainer.classList.add(messageData.role);
                
                // Create label
                const label = document.createElement('strong');
                label.textContent = prefix;
                messageContainer.appendChild(label);
                
                // Create message bubble
                const messageBubble = document.createElement('div');
                messageBubble.classList.add('message-bubble');
                messageBubble.innerHTML = messageData.content;
                
                messageContainer.appendChild(messageBubble);
                messageDiv.appendChild(messageContainer);
                messagesDiv.appendChild(messageDiv);
            });
            
            // Scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            if (messages.length > 0) {
                showNotification(`Loaded ${messages.length} previous messages`, 'success', 3000);
            }
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
        showNotification('Error loading chat history', 'error', 3000);
    }
}

function clearChatHistory() {
    try {
        localStorage.removeItem(CHAT_HISTORY_KEY);
        localStorage.removeItem(PENDING_REQUESTS_KEY);
        const messagesDiv = document.getElementById('chat-messages');
        const messages = messagesDiv.querySelectorAll('.message');
        messages.forEach(message => message.remove());
        
        // Restore welcome screen
        const welcomeHTML = `
            <div id="welcome-container">
                <img src="/img/logo.png" alt="Welcome Logo">
                <p id="welcome-text">${translations[currentLang].welcomeText}</p>
                <h1 id="acknowledgement">${translations[currentLang].acknowledgement}</h1>
                <p id="author">${translations[currentLang].author}</p>
                <a id="license" href="https://github.com/Lelekhoa1812/Medical-Chatbot-RAG/blob/main/LICENSE">${translations[currentLang].license}</a>
            </div>`;
        messagesDiv.innerHTML = welcomeHTML;
        
        showNotification('Chat history cleared', 'success', 2000);
    } catch (error) {
        console.error('Error clearing chat history:', error);
        showNotification('Error clearing chat history', 'error', 3000);
    }
}

// Pending request management functions
function savePendingRequest(requestId, query, timestamp) {
    try {
        const pendingRequests = getPendingRequests();
        pendingRequests.push({
            requestId,
            query,
            timestamp,
            status: 'pending'
        });
        
        // Keep only the most recent requests
        const limitedRequests = pendingRequests.slice(-MAX_PENDING_REQUESTS);
        localStorage.setItem(PENDING_REQUESTS_KEY, JSON.stringify(limitedRequests));
        console.log(`Saved pending request ${requestId}`);
    } catch (error) {
        console.error('Error saving pending request:', error);
    }
}

function getPendingRequests() {
    try {
        const saved = localStorage.getItem(PENDING_REQUESTS_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch (error) {
        console.error('Error getting pending requests:', error);
        return [];
    }
}

function removePendingRequest(requestId) {
    try {
        const pendingRequests = getPendingRequests();
        const filtered = pendingRequests.filter(req => req.requestId !== requestId);
        localStorage.setItem(PENDING_REQUESTS_KEY, JSON.stringify(filtered));
        console.log(`Removed pending request ${requestId}`);
    } catch (error) {
        console.error('Error removing pending request:', error);
    }
}

async function checkPendingRequests() {
    try {
        const pendingRequests = getPendingRequests();
        if (pendingRequests.length === 0) return;
        
        console.log(`Checking ${pendingRequests.length} pending requests...`);
        
        for (const request of pendingRequests) {
            try {
                const response = await fetch(`${API_PREFIX}/check-request/${request.requestId}`);
                const data = await response.json();
                
                if (data.status === 'completed' && data.response) {
                    // Found a completed response!
                    console.log(`Found completed response for request ${request.requestId}`);
                    
                    // Remove the loader message if it exists
                    removeLastMessage();
                    
                    // Add the bot response
                    const htmlResponse = marked.parse(data.response);
                    const processedResponse = processCitations(htmlResponse);
                    appendMessage('bot', processedResponse, true);
                    addCitationListeners();
                    
                    // Remove from pending requests
                    removePendingRequest(request.requestId);
                    
                    // Show notification
                    showNotification('Restored response from previous session', 'success', 3000);
                } else if (data.status === 'failed') {
                    // Request failed, remove from pending
                    console.log(`Request ${request.requestId} failed: ${data.error}`);
                    removePendingRequest(request.requestId);
                }
                // If still pending, keep it in the list
            } catch (error) {
                console.error(`Error checking request ${request.requestId}:`, error);
            }
        }
    } catch (error) {
        console.error('Error checking pending requests:', error);
    }
}

// Periodic check for pending requests (every 30 seconds)
function startPendingRequestChecker() {
    setInterval(async () => {
        const pendingRequests = getPendingRequests();
        if (pendingRequests.length > 0) {
            console.log('Periodic check: Found pending requests');
            await checkPendingRequests();
        }
    }, 30000); // Check every 30 seconds
}

// Cleanup old pending requests (older than 1 hour)
function cleanupOldPendingRequests() {
    try {
        const pendingRequests = getPendingRequests();
        const oneHourAgo = Date.now() - (60 * 60 * 1000);
        
        const validRequests = pendingRequests.filter(req => req.timestamp > oneHourAgo);
        
        if (validRequests.length !== pendingRequests.length) {
            localStorage.setItem(PENDING_REQUESTS_KEY, JSON.stringify(validRequests));
            console.log(`Cleaned up ${pendingRequests.length - validRequests.length} old pending requests`);
        }
    } catch (error) {
        console.error('Error cleaning up old pending requests:', error);
    }
}

// Video display functions
function displayVideos(videos) {
    if (!videos || videos.length === 0) return;
    
    const messagesDiv = document.getElementById('chat-messages');
    const videoContainer = document.createElement('div');
    videoContainer.classList.add('video-container');
    videoContainer.innerHTML = `
        <div class="video-header">
            <h4>Related Medical Videos</h4>
        </div>
        <div class="video-grid">
            ${videos.map((video, index) => createVideoCard(video, index)).join('')}
        </div>
    `;
    
    messagesDiv.appendChild(videoContainer);
    
    // Scroll to bottom
    messagesDiv.scrollTo({
        top: messagesDiv.scrollHeight,
        behavior: 'smooth'
    });
    
    // Add event listeners for video cards
    addVideoEventListeners();
}

function createVideoCard(video, index) {
    const videoId = `video-${index}`;
    const embedUrl = getEmbedUrl(video.url);
    
    return `
        <div class="video-card" data-video-id="${videoId}">
            <div class="video-thumbnail" onclick="toggleVideo('${videoId}')">
                <img src="${video.thumbnail}" alt="${video.title}" />
                <div class="video-overlay">
                    <i class="fas fa-play"></i>
                </div>
                <div class="video-duration">${video.duration || ''}</div>
            </div>
            <div class="video-info">
                <h5 class="video-title">${video.title}</h5>
                <p class="video-channel">${video.channel || video.source}</p>
                <div class="video-actions">
                    <button class="video-btn" onclick="toggleVideo('${videoId}')">
                        <i class="fas fa-play"></i> Watch
                    </button>
                    <a href="${video.url}" target="_blank" class="video-btn external">
                        <i class="fas fa-external-link-alt"></i> Open
                    </a>
                </div>
            </div>
            <div class="video-player" id="${videoId}" style="display: none;">
                <iframe 
                    src="${embedUrl}" 
                    frameborder="0" 
                    allowfullscreen
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
                </iframe>
                <button class="close-video" onclick="toggleVideo('${videoId}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
}

function getEmbedUrl(url) {
    // Convert YouTube URLs to embed format
    if (url.includes('youtube.com/watch')) {
        const videoId = url.split('v=')[1]?.split('&')[0];
        return videoId ? `https://www.youtube.com/embed/${videoId}` : url;
    } else if (url.includes('youtu.be/')) {
        const videoId = url.split('youtu.be/')[1]?.split('?')[0];
        return videoId ? `https://www.youtube.com/embed/${videoId}` : url;
    }
    // For other platforms, return original URL (they might not support embedding)
    return url;
}

function toggleVideo(videoId) {
    const videoPlayer = document.getElementById(videoId);
    const videoCard = videoPlayer.closest('.video-card');
    
    if (videoPlayer.style.display === 'none') {
        // Show video player
        videoPlayer.style.display = 'block';
        videoCard.classList.add('expanded');
        
        // Hide other expanded videos
        document.querySelectorAll('.video-card.expanded').forEach(card => {
            if (card !== videoCard) {
                const otherPlayer = card.querySelector('.video-player');
                otherPlayer.style.display = 'none';
                card.classList.remove('expanded');
            }
        });
        
        // Scroll to video
        videoCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
        // Hide video player
        videoPlayer.style.display = 'none';
        videoCard.classList.remove('expanded');
    }
}

function addVideoEventListeners() {
    // Event listeners are added via onclick attributes in the HTML
    // This function can be used for additional event handling if needed
    console.log('Video event listeners added');
}

// Translation strings
const translations = {
    "EN": {
    header: "Medical Chatbot Doctor",
    tooltip: "Hello, how can I help you today?",
    upload_tooltip: "Upload medical image diagnosis.",
    welcomeText: "Hi! I’m your dedicated health assistant, here to support you with all your wellness questions. Feel free to ask me any question about your health and well-being.",
    acknowledgement: "Acknowledgement",
    author: "Author: (Liam) Dang Khoa Le",
    license: "License: Apache 2.0 License",
    chatInputPlaceholder: "Type your question here...",
    you: "You",
    bot: "DocBot",
    account: "Account",
    subscription: "Subscription",
    about: "About",
    loaderMessage: "Doctor Chatbot is finding the best solution for you, hang tight..."
    },
    "VI": {
    header: "Bác Sĩ Chatbot",
    tooltip: "Xin chào, tôi có thể giúp gì cho bạn?",
    upload_tooltip: "Tải hình ảnh y tế chẩn đoán.",
    welcomeText: "Chào bạn! Tôi là trợ lý sức khỏe tận tâm của bạn, sẵn sàng hỗ trợ mọi thắc mắc về sức khỏe và phúc lợi của bạn. Hãy thoải mái đặt câu hỏi nhé!",
    acknowledgement: "Thông tin",
    author: "Tác giả: Lê Đăng Khoa",
    license: "Giấy phép: Apache 2.0",
    chatInputPlaceholder: "Nhập câu hỏi của bạn...",
    you: "Bạn",
    bot: "Bác Sĩ Chatbot",
    account: "Tài Khoản",
    subscription: "Đăng Ký",
    about: "Thông Tin",
    loaderMessage: "Bác sĩ Chatbot đang tìm giải pháp tốt nhất cho bạn, vui lòng chờ trong giây lát..."
    },
    "ZH": {
    header: "医疗聊天机器人医生",
    tooltip: "您好，我今天能为您提供什么帮助？",
    upload_tooltip: "上传您的医学图像以供诊断.",
    welcomeText: "您好！我是您专属的健康助手，随时为您解答关于健康与福祉的问题。请随时向我提问。",
    acknowledgement: "鸣谢",
    author: "作者：黎登科",
    license: "许可证：Apache 2.0 许可证",
    chatInputPlaceholder: "请输入您的问题...",
    you: "您",
    bot: "医生机器人",
    account: "账户",
    subscription: "订阅",
    about: "关于",
    loaderMessage: "医生聊天机器人正在为您寻找最佳解决方案，请稍候…"
    }
};

// Assign user_id on session
function getUserId() {
    let uid = localStorage.getItem('chat_user_id');
    if (!uid) {
      uid = crypto.randomUUID();
      localStorage.setItem('chat_user_id', uid);
    }
    return uid;
}

// Function to update all UI strings based on selected language
function updateLanguage(lang) {
    currentLang = lang;
    const t = translations[lang];
    // Update nav header and tooltip (guarded)
    const navHeader = document.getElementById('nav-header');
    if (navHeader) navHeader.innerText = t.header;
    const tooltip = document.getElementById('tooltip');
    if (tooltip) tooltip.innerText = t.tooltip;
    const uploadTooltip = document.getElementById('upload-tooltip');
    if (uploadTooltip) uploadTooltip.innerText = t.upload_tooltip;
    // Update chat header title text (target the inner title element)
    const headerTitle = document.querySelector('.header-title');
    if (headerTitle) headerTitle.innerText = t.header;
    // Update welcome screen texts (guarded)
    const welcomeText = document.getElementById('welcome-text');
    if (welcomeText) welcomeText.innerText = t.welcomeText;
    const acknowledgement = document.getElementById('acknowledgement');
    if (acknowledgement) acknowledgement.innerText = t.acknowledgement;
    const author = document.getElementById('author');
    if (author) author.innerText = t.author;
    const license = document.getElementById('license');
    if (license) license.innerText = t.license;
    // Update chat input placeholder (guarded)
    const userInput = document.getElementById('user-input');
    if (userInput) userInput.placeholder = t.chatInputPlaceholder;
    // Update nav links (both desktop and mobile)
    const accountLinks = document.querySelectorAll('#nav-account, #nav-account-mobile');
    const subscriptionLinks = document.querySelectorAll('#nav-subscription, #nav-subscription-mobile');
    const aboutLinks = document.querySelectorAll('#nav-about, #nav-about-mobile');
    accountLinks.forEach(link => link.innerText = t.account);
    subscriptionLinks.forEach(link => link.innerText = t.subscription);
    aboutLinks.forEach(link => link.innerText = t.about);
}

// Function to toggle theme
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    
    // Update theme toggle icon
    const themeToggle = document.getElementById('theme-toggle');
    const icon = themeToggle.querySelector('i');
    if (currentTheme === 'dark') {
        icon.className = 'fas fa-sun';
    } else {
        icon.className = 'fas fa-moon';
    }
}

// Function to toggle input mode independently
function toggleInputMode(mode) {
    const modeButton = document.getElementById(`${mode}-mode-btn`);
    const uploadLabel = document.getElementById('upload-label');
    
    if (mode === 'search') {
        searchModeActive = !searchModeActive;
        modeButton.classList.toggle('active', searchModeActive);
        modeButton.classList.toggle('inactive', !searchModeActive);
    } else if (mode === 'upload') {
        uploadModeActive = !uploadModeActive;
        modeButton.classList.toggle('active', uploadModeActive);
        modeButton.classList.toggle('inactive', !uploadModeActive);
        
        // Show/hide upload icon based on upload mode
        if (uploadModeActive) {
            uploadLabel.style.display = 'flex';
        } else {
            uploadLabel.style.display = 'none';
        }
    } else if (mode === 'video') {
        videoModeActive = !videoModeActive;
        modeButton.classList.toggle('active', videoModeActive);
        modeButton.classList.toggle('inactive', !videoModeActive);
    }
    
    // Update UI feedback
    updateModeFeedback();
}

// Function to update mode feedback
function updateModeFeedback() {
    const searchBtn = document.getElementById('search-mode-btn');
    const uploadBtn = document.getElementById('upload-mode-btn');
    const videoBtn = document.getElementById('video-mode-btn');
    const inputModes = document.querySelector('.input-modes');
    
    // Update button states
    searchBtn.classList.toggle('active', searchModeActive);
    searchBtn.classList.toggle('inactive', !searchModeActive);
    uploadBtn.classList.toggle('active', uploadModeActive);
    uploadBtn.classList.toggle('inactive', !uploadModeActive);
    videoBtn.classList.toggle('active', videoModeActive);
    videoBtn.classList.toggle('inactive', !videoModeActive);
    
    // Count active modes
    const activeModes = [searchModeActive, uploadModeActive, videoModeActive].filter(Boolean).length;
    
    // Add special class when multiple modes are active
    inputModes.classList.toggle('multiple-active', activeModes > 1);
    
    // Show notification about current mode state
    if (activeModes === 0) {
        showNotification('No modes selected - text input only', 'warning', 3000);
    } else if (activeModes === 1) {
        if (searchModeActive) {
            showNotification('Search mode active', 'success', 2000);
        } else if (uploadModeActive) {
            showNotification('Upload mode active', 'success', 2000);
        } else if (videoModeActive) {
            showNotification('Video mode active', 'success', 2000);
        }
    } else {
        const activeModeNames = [];
        if (searchModeActive) activeModeNames.push('Search');
        if (uploadModeActive) activeModeNames.push('Upload');
        if (videoModeActive) activeModeNames.push('Video');
        showNotification(`${activeModeNames.join(', ')} modes are active`, 'info', 3000);
    }
}

// Initialize theme from localStorage
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    currentTheme = savedTheme;
    document.documentElement.setAttribute('data-theme', currentTheme);
    
    // Update theme toggle icon
    const themeToggle = document.getElementById('theme-toggle');
    const icon = themeToggle.querySelector('i');
    if (currentTheme === 'dark') {
        icon.className = 'fas fa-sun';
    } else {
        icon.className = 'fas fa-moon';
    }
}

// Initialize mobile menu state
function initializeMobileMenu() {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    if (!hamburger || !mobileNav) return;
    
    // Ensure menu is closed on initialization
    hamburger.classList.remove('active');
    mobileNav.classList.remove('active');
    document.body.classList.remove('menu-open');
    
    // Check if we're on desktop and ensure mobile menu is hidden
    if (window.innerWidth > 768) {
        closeMobileMenu();
    }
}

// Notification System
function showNotification(message, type = 'info', duration = 5000) {
    const toast = document.getElementById('notification-toast');
    const icon = toast.querySelector('.notification-icon');
    const messageEl = toast.querySelector('.notification-message');
    
    // Set icon based on type
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    
    icon.className = `notification-icon ${icons[type] || icons.info}`;
    messageEl.textContent = message;
    
    // Show notification
    toast.classList.remove('hidden');
    
    // Auto hide after duration
    setTimeout(() => {
        hideNotification();
    }, duration);
}

function hideNotification() {
    const toast = document.getElementById('notification-toast');
    toast.classList.add('hidden');
}

// Settings functionality
function openSettings() {
    showNotification('Settings panel coming soon!', 'info');
}

// Hamburger Menu functionality
function toggleMobileMenu(event) {
    event.preventDefault();
    event.stopPropagation();
    
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    if (!hamburger || !mobileNav) return;
    
    const isActive = hamburger.classList.contains('active');
    
    if (isActive) {
        closeMobileMenu();
    } else {
        openMobileMenu();
    }
}

// Open mobile menu
function openMobileMenu() {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    if (!hamburger || !mobileNav) return;
    
    hamburger.classList.add('active');
    mobileNav.classList.add('active');
    
    // Add body class to prevent scrolling
    document.body.classList.add('menu-open');
}

// Close mobile menu
function closeMobileMenu() {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    if (!hamburger || !mobileNav) return;
    
    hamburger.classList.remove('active');
    mobileNav.classList.remove('active');
    
    // Remove body class to allow scrolling
    document.body.classList.remove('menu-open');
}

// Check if click is outside mobile menu
function isClickOutsideMobileMenu(event) {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    if (!hamburger || !mobileNav) return true;
    
    return !hamburger.contains(event.target) && !mobileNav.contains(event.target);
}

// --- Remove last message ---
function removeLastMessage() {
    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv.lastChild) {
        messagesDiv.removeChild(messagesDiv.lastChild);
    }
}

// --- Process citations/links and render as compact domain buttons ---
function processCitations(htmlContent) {
    // First, clean up malformed citation tags
    let cleanedContent = htmlContent;
    
    // Fix malformed citations like <https://example.com> <##2> or <https://example.com#1>
    cleanedContent = cleanedContent.replace(/<https?:\/\/[^>]*>[\s]*<##?\d+>/g, (match) => {
        // Extract the URL part
        const urlMatch = match.match(/<https?:\/\/[^>]*>/);
        return urlMatch ? urlMatch[0] : match;
    });
    
    // Fix citations with hash fragments like <https://example.com#1>
    cleanedContent = cleanedContent.replace(/<https?:\/\/[^>]*#\d+>/g, (match) => {
        // Remove the hash fragment
        return match.replace(/#\d+>/, '>');
    });
    
    // Fix citations with malformed hash tags like <https://example.com> <##2>
    cleanedContent = cleanedContent.replace(/<https?:\/\/[^>]*>[\s]*<##?\d+>/g, (match) => {
        const urlMatch = match.match(/<https?:\/\/[^>]*>/);
        return urlMatch ? urlMatch[0] : match;
    });
    
    // 1) Process <https://...> style citations
    const citationPattern = /<https?:\/\/[^>]+>/g;
    cleanedContent = cleanedContent.replace(citationPattern, (match) => {
        const url = match.slice(1, -1); // Remove < and >
        const domain = extractDomain(url);
        return `<span class="citation-link" data-url="${url}" title="View source: ${domain}">
                    <i class="fas fa-external-link-alt citation-icon"></i>
                    <span class="citation-domain">${domain}</span>
                </span>`;
    });

    // 2) Process anchor tags created by markdown (<a href="...">...</a>)
    // Replace anchor content with compact domain button, preserve original href as data-url
    const anchorPattern = /<a\s+href=\"(https?:\/\/[^\"\s]+)\"[^>]*>([\s\S]*?)<\/a>/gi;
    cleanedContent = cleanedContent.replace(anchorPattern, (full, href, inner) => {
        // Ignore anchors that already look like our citation-link
        if (/class=\"citation-link\"/.test(full)) return full;
        const domain = extractDomain(href);
        return `<span class="citation-link" data-url="${href}" title="View source: ${domain}">
                    <i class="fas fa-external-link-alt citation-icon"></i>
                    <span class="citation-domain">${domain}</span>
                </span>`;
    });

    return cleanedContent;
}

// --- Extract domain from URL for display ---
function extractDomain(url) {
    try {
        const urlObj = new URL(url);
        return urlObj.hostname.replace('www.', '');
    } catch (e) {
        return 'Source';
    }
}

// --- Add event listeners for citation links ---
function addCitationListeners() {
    const citationLinks = document.querySelectorAll('.citation-link');
    citationLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const url = this.getAttribute('data-url');
            if (url) {
                window.open(url, '_blank');
            }
        });
    });
} 

// Stack over
let pendingImageBase64 = null;
let pendingImageDesc = null;

// --- Send message over ---
async function sendMessage(customQuery = null, imageBase64 = null) {
    // Prevent duplicate submissions
    if (isSubmitting) {
        console.log('Already submitting, ignoring duplicate request');
        return;
    }
    
    // Debounce rapid successive submissions
    const now = Date.now();
    if (now - lastSubmissionTime < SUBMISSION_DEBOUNCE_MS) {
        console.log('Submission too soon, debouncing');
        return;
    }
    
    const user_id = getUserId();
    const input = document.getElementById('user-input');
    const message = customQuery || input.value.trim();
    
    // Handle empty message properly
    if (!message) {
        if (!pendingImageDesc) {
            alert("Empty Message!");
            isSubmitting = false; // Reset submission state
            return; // Add return to prevent further execution
        } else {
            // Use pending image description if no text message
            message = pendingImageDesc;
        }
    }
    
    // Set submission state
    isSubmitting = true;
    lastSubmissionTime = now;
    
    // Disable send button during submission
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.style.opacity = '0.6';
    } 
    // Remove welcome screen if shown
    const welcomeContainer = document.getElementById('welcome-container');
    if (welcomeContainer) welcomeContainer.remove();
    // Add user message
    appendMessage('user', message, false);
    if (!customQuery) input.value = '';
    // Add loader
    const loaderHTML = `<div class="loader-container"><div class="loader"></div><div class="loader-text">${translations[currentLang].loaderMessage}</div></div>`;
    appendMessage('bot', loaderHTML, true);
    // Prepare JSON body
    const body = {
        query: message,
        lang: currentLang,
        user_id,
        search: !!searchModeActive,
        video: !!videoModeActive,
        ...(pendingImageBase64 ? { image_base64: pendingImageBase64 } : {}),
        img_desc: pendingImageDesc,
    };
    // Send over backend
    try {
        const response = await fetch(`${API_PREFIX}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await response.json();
        
        // Store request ID for persistence
        if (data.request_id) {
            savePendingRequest(data.request_id, message, Date.now());
        }
        
        // Remove message and img previewer
        removeLastMessage();
        pendingImageBase64 = null;
        pendingImageDesc = null;
        const previewEl = document.getElementById('upload-preview-container');
        if (previewEl) previewEl.remove();
        
        // Configure Marked.js to handle nested formatting properly
        // Using standard Marked.js parsing with enhanced CSS for styling
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
        
        // Parse markdown and let CSS handle the styling
        // This approach avoids conflicts with Marked.js internals
        let htmlResponse = marked.parse(data.response);
        
        // Process citation tags and replace with magnifier icons
        htmlResponse = processCitations(htmlResponse);
        
        // Debug: Log the parsed HTML to see what's generated
        console.log('🔍 Parsed HTML:', htmlResponse);
        console.log('🔍 Original response:', data.response);
        
        appendMessage('bot', htmlResponse, true);
        
        // Add event listeners for citation links
        addCitationListeners();
        
        // Handle video data if present
        if (data.videos && data.videos.length > 0) {
            displayVideos(data.videos);
        }
        
        // Remove from pending requests since we got the response
        if (data.request_id) {
            removePendingRequest(data.request_id);
        }
    } catch (err) {
        removeLastMessage();
        appendMessage('bot', "❌ Failed to get a response. Please try again.", false);
        console.error(err);
    } finally {
        // Always reset submission state and re-enable send button
        isSubmitting = false;
        const sendBtn = document.getElementById('send-btn');
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.style.opacity = '1';
        }
    }
}

// --- Render msg over ---
function appendMessage(role, text, isHTML) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    
    // Create message structure
    const prefix = role === 'user' ? translations[currentLang].you : translations[currentLang].bot;
    
    // Create message container
    const messageContainer = document.createElement('div');
    messageContainer.classList.add(role);
    
    // Create label
    const label = document.createElement('strong');
    label.textContent = prefix;
    messageContainer.appendChild(label);
    
    // Create message bubble
    const messageBubble = document.createElement('div');
    messageBubble.classList.add('message-bubble');
    
    // Set content
    if (isHTML) {
        messageBubble.innerHTML = text;
    } else {
        messageBubble.textContent = text;
    }
    
    // If this is a user message and pendingImageBase64 is set, include image preview
    if (role === 'user' && pendingImageBase64) {
        const imagePreview = document.createElement('div');
        imagePreview.className = 'chat-preview-image-block';
        imagePreview.innerHTML = `
                <img src="data:image/jpeg;base64,${pendingImageBase64}" alt="User Image" />
                <p class="image-desc">${pendingImageDesc}</p>
        `;
        messageBubble.appendChild(imagePreview);
    }
    
    messageContainer.appendChild(messageBubble);
    messageDiv.appendChild(messageContainer);
    messagesDiv.appendChild(messageDiv);
    
    // Smooth scroll to bottom
    messagesDiv.scrollTo({
        top: messagesDiv.scrollHeight,
        behavior: 'smooth'
    });
    
    // Save chat history after adding message
    saveChatHistory();
}


// --- Dropdown Lang Selector ---
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    initializeTheme();
    
    // Load chat history on page load
    loadChatHistory();
    
    // Check for pending requests from previous session
    checkPendingRequests();
    
    // Start periodic checker for pending requests
    startPendingRequestChecker();
    
    // Cleanup old pending requests
    cleanupOldPendingRequests();
    
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    themeToggle.addEventListener('click', toggleTheme);
    
    // Clear history button functionality
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear all chat history? This action cannot be undone.')) {
                clearChatHistory();
            }
        });
    }
    
    // Settings button functionality
    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', openSettings);
    }
    
    // Notification close functionality
    const notificationClose = document.querySelector('.notification-close');
    if (notificationClose) {
        notificationClose.addEventListener('click', hideNotification);
    }
    
    // Hamburger menu functionality
    const hamburgerMenu = document.getElementById('hamburger-menu');
    if (hamburgerMenu) {
        hamburgerMenu.addEventListener('click', toggleMobileMenu);
    }
    
    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
        if (isClickOutsideMobileMenu(event)) {
            closeMobileMenu();
        }
    });
    
    // Close mobile menu when clicking on mobile nav links
    const mobileNavLinks = document.querySelectorAll('.mobile-nav a');
    mobileNavLinks.forEach(link => {
        link.addEventListener('click', function() {
            closeMobileMenu();
        });
    });
    
    // Close mobile menu on window resize to desktop
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            closeMobileMenu();
        }
    });
    
    // Input mode functionality - independent toggles
    const searchModeBtn = document.getElementById('search-mode-btn');
    const uploadModeBtn = document.getElementById('upload-mode-btn');
    const videoModeBtn = document.getElementById('video-mode-btn');
    
    if (searchModeBtn) {
        searchModeBtn.addEventListener('click', function() {
            toggleInputMode('search');
        });
    }
    
    if (uploadModeBtn) {
        uploadModeBtn.addEventListener('click', function() {
            toggleInputMode('upload');
        });
    }
    
    if (videoModeBtn) {
        videoModeBtn.addEventListener('click', function() {
            toggleInputMode('video');
        });
    }
    
    const dropdownBtn = document.querySelector('.dropdown-btn');
    const dropdownMenu = document.querySelector('.dropdown-menu');
    dropdownBtn.addEventListener('click', function(event) {
        event.stopPropagation();
        dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
    });

    // --- Dropdown Lang Selection ---
    document.querySelectorAll('.dropdown-menu li').forEach(item => {
        item.addEventListener('click', function(event) {
            event.stopPropagation();
            const selectedLang = this.getAttribute('data-lang');
            dropdownBtn.innerHTML = selectedLang + " &#x25BC;";
            dropdownMenu.style.display = 'none';
            updateLanguage(selectedLang);
        });
    });
    // Close the dropdown if clicking outside
    document.addEventListener('click', function() {dropdownMenu.style.display = 'none';});

    // --- Trigger message sender ---
    // 1. By btn click
    const sendBtn = document.getElementById('send-btn');
    sendBtn.addEventListener('click', () => {
        // Only send if not already submitting
        if (!isSubmitting) {
            sendMessage();
        }
    });
    // 2. By enter key-press
    document.getElementById("user-input").addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault(); // prevent newline
          // Only send if not already submitting
          if (!isSubmitting) {
          sendMessage(); // your custom send function
          }
        }
    });
    
    // --- Language modal ---
    const modal = document.getElementById('language-modal');
    const modalButtons = modal.querySelectorAll('button');
    // When any modal button is clicked:
    modalButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const lang = this.getAttribute('data-lang');
            updateLanguage(lang);
            // Also update the dropdown button text
            document.querySelector('.dropdown-btn').innerHTML = lang + " &#x25BC;";
            // Hide the modal
            modal.style.display = 'none';
        });
    });

    // --- Image Upload Flow ---
    const uploadInput = document.getElementById('image-upload');
    // Show img preview on desc modal
    uploadInput.addEventListener('change', function () {
        const file = this.files[0];
        if (!file) return;
        // Init
        const modal = document.getElementById('image-modal');
        const preview = document.getElementById('uploaded-preview');
        const reader = new FileReader();
        // Loader
        reader.onload = function (e) {
          preview.src = e.target.result;
          modal.classList.remove('hidden'); // Show modal
        };
        reader.readAsDataURL(file); // Convert image to base64 for preview
      });
    let uploadedImageFile = null;
    // Init
    document.getElementById('image-upload').addEventListener('change', function () {
        const file = this.files[0];
        if (!file) return;
        uploadedImageFile = file;
        document.getElementById('image-modal').classList.remove('hidden');
        this.value = '';
    });
    // Cancel modal
    document.getElementById('cancel-image-modal').addEventListener('click', () => {
        uploadedImageFile = null;
        document.getElementById('image-description').value = '';
        document.getElementById('image-modal').classList.add('hidden');
    });
    // Submit over
    document.getElementById('submit-image-modal').addEventListener('click', () => {
        const desc = document.getElementById('image-description').value.trim() || 
                     "Describe and investigate any clinical findings from this medical image.";
        const file = uploadedImageFile;
        if (!file) return;
        // Read img
        const reader = new FileReader();
        reader.onload = function (e) {
            pendingImageBase64 = e.target.result.split(',')[1];
            pendingImageDesc = desc;
            // Add preview container just above the input box
            const previewHTML = `
              <div id="upload-preview-container">
                <div class="image-preview-block">
                  <img src="${e.target.result}" alt="Preview" />
                  <p>${desc}</p>
                  <button id="remove-preview">✖</button>
                </div>
              </div>`;
            document.querySelector('.chat-input').insertAdjacentHTML('afterbegin', previewHTML);
            // Cleanup
            document.getElementById('image-modal').classList.add('hidden');
            uploadedImageFile = null;
            document.getElementById('image-description').value = '';
            // Handle preview removal
            document.getElementById('remove-preview').addEventListener('click', () => {
                document.getElementById('upload-preview-container').remove();
                pendingImageBase64 = null;
                pendingImageDesc = null;
            });
        };
        reader.readAsDataURL(file);
    });    
});