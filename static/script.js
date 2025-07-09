// Render app API prefix
// Decide the API endpoint based on local dev or production
const isLocal = window.location.hostname === "localhost";
const API_PREFIX = isLocal
  ? "http://0.0.0.0:8000"  // local dev
//   : "https://my-medical-chatbot.onrender.com";     // production Render server
//   : "https://medical-chatbot-henna.streamlit.app"; // production Streamlit server
     : "https://BinKhoaLe1812-Medical-Chatbot.hf.space"


console.log(marked.parse("### **Important Message**"));

// Global variable for current language (default English)
let currentLang = "EN";

// Translation strings
const translations = {
    "EN": {
    header: "Medical Chatbot Doctor",
    tooltip: "Hello, how can I help you today?",
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

// Remove last message (for loader)
function removeLastMessage() {
    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv.lastChild) {
        messagesDiv.removeChild(messagesDiv.lastChild);
    }
} 

// Send the message to server-side
async function sendMessage() {
    const user_id = getUserId();
    const input = document.getElementById('user-input');
    const message = input.value;
    if (!message) return;
    // Remove welcome screen if exists
        const welcomeContainer = document.getElementById('welcome-container');
    if (welcomeContainer) {
        welcomeContainer.remove();
    }
    appendMessage('user', message, false);
    input.value = '';
    // Insert loader message as bot message
    const loaderHTML = `<div class="loader-container"><div class="loader"></div><div class="loader-text">${translations[currentLang].loaderMessage}</div></div>`;
    appendMessage('bot', loaderHTML, true);

    // Post the query (and language) to the backend
    const response = await fetch(`${API_PREFIX}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message, lang: currentLang, user_id })
    });
    const data = await response.json();
    const htmlResponse = marked.parse(data.response);
    removeLastMessage();
    appendMessage('bot', htmlResponse, true);
}

function appendMessage(role, text, isHTML) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    const prefix = role === 'user' ? translations[currentLang].you : translations[currentLang].bot;
    if (isHTML) {
        messageDiv.innerHTML = `<strong class="${role}">${prefix}:</strong><br/>${text}`;
    } else {
        messageDiv.innerHTML = `<strong class="${role}">${prefix}:</strong> ${text}`;
    }
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Dropdown language selector functionality
document.addEventListener('DOMContentLoaded', function() {
    const dropdownBtn = document.querySelector('.dropdown-btn');
    const dropdownMenu = document.querySelector('.dropdown-menu');
    dropdownBtn.addEventListener('click', function(event) {
        event.stopPropagation();
        dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
    });

    // When a language option is selected from the dropdown
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
    document.addEventListener('click', function() {
        dropdownMenu.style.display = 'none';
    });

    // Trigger message sender
    // 1. By btn click
    const sendBtn = document.getElementById('send-btn');
    sendBtn.addEventListener('click', sendMessage);
    // 2. By enter key-press
    document.getElementById("user-input").addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault(); // prevent newline
          sendMessage(); // your custom send function
        }
      });

    // Handle image upload
    const uploadInput = document.getElementById('image-upload');
    uploadInput.addEventListener('change', async function () {
    const user_id = getUserId();
    const file = this.files[0];
    if (!file) return;
    // Append loader
    appendMessage('user', "📷 Uploaded an image for diagnosis.", false);
    const loaderHTML = `<div class="loader-container"><div class="loader"></div><div class="loader-text">${translations[currentLang].loaderMessage}</div></div>`;
    appendMessage('bot', loaderHTML, true);
    // Append data
    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('user_id', user_id);
        formData.append('lang', currentLang);
        // Send over
        const response = await fetch(`${API_PREFIX}/image`, {
        method: 'POST',
        body: formData
        });
        // Await for response
        const data = await response.json();
        removeLastMessage();
        const htmlResponse = marked.parse(data.response);
        appendMessage('bot', htmlResponse, true);
    } catch (error) {
        removeLastMessage();
        appendMessage('bot', "❌ Failed to process image. Please try again later.", false);
        console.error(error);
    }
    // Clear file after upload
    this.value = '';
});

});

// Modal Language Selection Functionality
document.addEventListener('DOMContentLoaded', function() {
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
});