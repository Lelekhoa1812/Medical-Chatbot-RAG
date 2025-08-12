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
    // Update nav links
    document.getElementById('nav-account').innerText = translations[lang].account;
    document.getElementById('nav-subscription').innerText = translations[lang].subscription;
    document.getElementById('nav-about').innerText = translations[lang].about;
}

// --- Remove last message ---
function removeLastMessage() {
    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv.lastChild) {
        messagesDiv.removeChild(messagesDiv.lastChild);
    }
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
        const htmlResponse = marked.parse(data.response);
        appendMessage('bot', htmlResponse, true);
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
    // Append components
    messageDiv.innerHTML = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}


// --- Dropdown Lang Selector ---
document.addEventListener('DOMContentLoaded', function() {
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