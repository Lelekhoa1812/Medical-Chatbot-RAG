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
    // Update nav header and tooltip
    document.getElementById('nav-header').innerText = translations[lang].header;
    document.getElementById('tooltip').innerText = translations[lang].tooltip;
    document.getElementById('upload-tooltip').innerText = translations[lang].upload_tooltip;
    // Update chat header
    document.getElementById('chat-header').innerText = translations[lang].header;
    // Update welcome screen texts
    document.getElementById('welcome-text').innerText = translations[lang].welcomeText;
    document.getElementById('acknowledgement').innerText = translations[lang].acknowledgement;
    document.getElementById('author').innerText = translations[lang].author;
    document.getElementById('license').innerText = translations[lang].license;
    // Update chat input placeholder
    document.getElementById('user-input').placeholder = translations[lang].chatInputPlaceholder;
    // Update nav links (both desktop and mobile)
    const accountLinks = document.querySelectorAll('#nav-account, #nav-account-mobile');
    const subscriptionLinks = document.querySelectorAll('#nav-subscription, #nav-subscription-mobile');
    const aboutLinks = document.querySelectorAll('#nav-about, #nav-about-mobile');
    
    accountLinks.forEach(link => link.innerText = translations[lang].account);
    subscriptionLinks.forEach(link => link.innerText = translations[lang].subscription);
    aboutLinks.forEach(link => link.innerText = translations[lang].about);
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
    }
    
    // Update UI feedback
    updateModeFeedback();
}

// Function to update mode feedback
function updateModeFeedback() {
    const searchBtn = document.getElementById('search-mode-btn');
    const uploadBtn = document.getElementById('upload-mode-btn');
    const inputModes = document.querySelector('.input-modes');
    
    // Update button states
    searchBtn.classList.toggle('active', searchModeActive);
    searchBtn.classList.toggle('inactive', !searchModeActive);
    uploadBtn.classList.toggle('active', uploadModeActive);
    uploadBtn.classList.toggle('inactive', !uploadModeActive);
    
    // Add special class when both modes are active
    inputModes.classList.toggle('both-active', searchModeActive && uploadModeActive);
    
    // Show notification about current mode state
    if (searchModeActive && uploadModeActive) {
        showNotification('Both Search and Upload modes are active', 'info', 3000);
    } else if (!searchModeActive && !uploadModeActive) {
        showNotification('No modes selected - text input only', 'warning', 3000);
    } else if (searchModeActive) {
        showNotification('Search mode active', 'success', 2000);
    } else if (uploadModeActive) {
        showNotification('Upload mode active', 'success', 2000);
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

// --- Process citations and replace with magnifier icons ---
function processCitations(htmlContent) {
    // Find all citation tags like <https://example.com> and replace with magnifier icons
    const citationPattern = /<https?:\/\/[^>]+>/g;
    
    return htmlContent.replace(citationPattern, (match) => {
        const url = match.slice(1, -1); // Remove < and >
        const domain = extractDomain(url);
        return `<span class="citation-link" data-url="${url}" title="View source: ${domain}">
                    <i class="fas fa-search-plus citation-icon"></i>
                    <span class="citation-domain">${domain}</span>
                </span>`;
    });
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
    const user_id = getUserId();
    const input = document.getElementById('user-input');
    const message = customQuery || input.value.trim();
    if (!message) {
        if (!pendingImageDesc) {
            alert("Empty Message!")
        }
        else {
            message = pendingImageDesc;
        }
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
        search: currentMode === "search",
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
    } catch (err) {
        removeLastMessage();
        appendMessage('bot', "❌ Failed to get a response. Please try again.", false);
        console.error(err);
    }
}

// --- Render msg over ---
function appendMessage(role, text, isHTML) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    // Prefixing
    const prefix = role === 'user' ? translations[currentLang].you : translations[currentLang].bot;
    // MarkDown -> HTML
    let content = '';
    if (isHTML) {
        content = `<strong class="${role}">${prefix}:</strong><br/>${text}`;
    } else {
        content = `<strong class="${role}">${prefix}:</strong> ${text}`;
    }
    // If this is a user message and pendingImageBase64 is set, include image preview
    if (role === 'user' && pendingImageBase64) {
        content += `
            <div class="chat-preview-image-block">
                <img src="data:image/jpeg;base64,${pendingImageBase64}" alt="User Image" />
                <p class="image-desc">${pendingImageDesc}</p>
            </div>`;
    }
    // Debug: Log the content being inserted
    if (role === 'bot') {
        console.log('🔍 Bot message content:', content);
        console.log('🔍 Message div element:', messageDiv);
    }
    
    // Append components
    messageDiv.innerHTML = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}


// --- Dropdown Lang Selector ---
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    initializeTheme();
    
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    themeToggle.addEventListener('click', toggleTheme);
    
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
    sendBtn.addEventListener('click', () => sendMessage());
    // 2. By enter key-press
    document.getElementById("user-input").addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault(); // prevent newline
          sendMessage(); // your custom send function
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