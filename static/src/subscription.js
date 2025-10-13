// Global variable for current language (default English)
let currentLang = "EN";

// Global variable for current theme (default light)
let currentTheme = "light";

// Translation strings for subscription page
const translations = {
    "EN": {
        // Navigation
        navAccount: "Back to Chat",
        navSubscription: "Support",
        navAbout: "About",
        
        // Hero Section
        heroTitle: "Support Medical AI Research",
        heroSubtitle: "Join our mission to put humanity back at the heart of healthcare",
        
        // Mission Section
        missionTitle: "Our Mission",
        missionText1: "We're building medical AI that understands patient history - because context saves lives.",
        missionText2: "As a non-profit SwinBurne initiative, we're creating AI that reads between the lines of medical data. Our technology integrates your full health story to support doctors in delivering truly personalized care.",
        missionText3: "This isn't for profit - it's for people. We're developing open solutions that will benefit everyone, not shareholders.",
        missionHighlight: "Your support fuels research that puts humanity back at the heart of healthcare.",
        missionCta: "Join our mission. Fund AI that cares.",
        
        // Donation Section
        donationTitle: "Support Our Research",
        donationSubtitle: "Help us continue developing life-saving medical AI technology",
        coffeeAmount: "Buy us a coffee",
        developmentAmount: "Support development",
        researchAmount: "Fund research",
        livesAmount: "Save lives",
        customAmountLabel: "Custom Amount:",
        customAmountPlaceholder: "Enter amount",
        donateButton: "Donate",
        secureNote: "Secure donations powered by Buy Me a Coffee",
        researcherName: "Dang Khoa Le (Liam)",
        researcherTitle: "Lead Researcher & Developer",
        institution: "Swinburne University of Technology",
        
        // Impact Section
        impactTitle: "Your Impact",
        impactOpenSource: "Open Source",
        impactOpenSourceDesc: "All our research and tools are freely available to benefit the global medical community",
        impactResearch: "Academic Research",
        impactResearchDesc: "Supporting evidence-based medical AI development at Swinburne University",
        impactGlobal: "Global Impact",
        impactGlobalDesc: "Creating technology that improves healthcare outcomes worldwide",
        
        // Footer
        footerCopyright: "© 2024 Medical Chatbot Research Team. All rights reserved.",
        footerInstitution: "Part of Swinburne University of Technology's AI Research Initiative"
    },
    "VI": {
        // Navigation
        navAccount: "Quay lại Chat",
        navSubscription: "Hỗ trợ",
        navAbout: "Giới thiệu",
        
        // Hero Section
        heroTitle: "Hỗ trợ Nghiên cứu AI Y tế",
        heroSubtitle: "Tham gia sứ mệnh đưa nhân văn trở lại trung tâm của chăm sóc sức khỏe",
        
        // Mission Section
        missionTitle: "Sứ mệnh của chúng tôi",
        missionText1: "Chúng tôi đang xây dựng AI y tế hiểu được lịch sử bệnh nhân - vì bối cảnh cứu sống con người.",
        missionText2: "Là một sáng kiến phi lợi nhuận của SwinBurne, chúng tôi đang tạo ra AI đọc được giữa các dòng dữ liệu y tế. Công nghệ của chúng tôi tích hợp toàn bộ câu chuyện sức khỏe của bạn để hỗ trợ bác sĩ cung cấp dịch vụ chăm sóc thực sự cá nhân hóa.",
        missionText3: "Đây không phải vì lợi nhuận - mà vì con người. Chúng tôi đang phát triển các giải pháp mở sẽ mang lại lợi ích cho tất cả mọi người, không phải cổ đông.",
        missionHighlight: "Sự hỗ trợ của bạn thúc đẩy nghiên cứu đưa nhân văn trở lại trung tâm của chăm sóc sức khỏe.",
        missionCta: "Tham gia sứ mệnh của chúng tôi. Tài trợ cho AI có tâm.",
        
        // Donation Section
        donationTitle: "Hỗ trợ Nghiên cứu của chúng tôi",
        donationSubtitle: "Giúp chúng tôi tiếp tục phát triển công nghệ AI y tế cứu sống",
        coffeeAmount: "Mua cà phê cho chúng tôi",
        developmentAmount: "Hỗ trợ phát triển",
        researchAmount: "Tài trợ nghiên cứu",
        livesAmount: "Cứu sống",
        customAmountLabel: "Số tiền tùy chỉnh:",
        customAmountPlaceholder: "Nhập số tiền",
        donateButton: "Quyên góp",
        secureNote: "Quyên góp an toàn được hỗ trợ bởi Buy Me a Coffee",
        researcherName: "Đặng Khoa Lê (Liam)",
        researcherTitle: "Nghiên cứu viên chính & Nhà phát triển",
        institution: "Đại học Công nghệ Swinburne",
        
        // Impact Section
        impactTitle: "Tác động của bạn",
        impactOpenSource: "Mã nguồn mở",
        impactOpenSourceDesc: "Tất cả nghiên cứu và công cụ của chúng tôi đều miễn phí để mang lại lợi ích cho cộng đồng y tế toàn cầu",
        impactResearch: "Nghiên cứu Học thuật",
        impactResearchDesc: "Hỗ trợ phát triển AI y tế dựa trên bằng chứng tại Đại học Swinburne",
        impactGlobal: "Tác động Toàn cầu",
        impactGlobalDesc: "Tạo ra công nghệ cải thiện kết quả chăm sóc sức khỏe trên toàn thế giới",
        
        // Footer
        footerCopyright: "© 2024 Nhóm Nghiên cứu Medical Chatbot. Tất cả quyền được bảo lưu.",
        footerInstitution: "Một phần của Sáng kiến Nghiên cứu AI của Đại học Công nghệ Swinburne"
    },
    "ZH": {
        // Navigation
        navAccount: "返回聊天",
        navSubscription: "支持",
        navAbout: "关于",
        
        // Hero Section
        heroTitle: "支持医疗AI研究",
        heroSubtitle: "加入我们的使命，将人文关怀重新置于医疗保健的核心",
        
        // Mission Section
        missionTitle: "我们的使命",
        missionText1: "我们正在构建理解患者病史的医疗AI - 因为背景信息拯救生命。",
        missionText2: "作为SwinBurne的非营利倡议，我们正在创建能够解读医疗数据深层含义的AI。我们的技术整合您的完整健康故事，支持医生提供真正个性化的护理。",
        missionText3: "这不是为了利润 - 而是为了人民。我们正在开发将使每个人受益的开放解决方案，而不是股东。",
        missionHighlight: "您的支持推动着将人文关怀重新置于医疗保健核心的研究。",
        missionCta: "加入我们的使命。资助有温度的AI。",
        
        // Donation Section
        donationTitle: "支持我们的研究",
        donationSubtitle: "帮助我们继续开发拯救生命的医疗AI技术",
        coffeeAmount: "请我们喝咖啡",
        developmentAmount: "支持开发",
        researchAmount: "资助研究",
        livesAmount: "拯救生命",
        customAmountLabel: "自定义金额：",
        customAmountPlaceholder: "输入金额",
        donateButton: "捐赠",
        secureNote: "由Buy Me a Coffee提供安全捐赠服务",
        researcherName: "黎光科 (Liam)",
        researcherTitle: "首席研究员兼开发者",
        institution: "斯威本科技大学",
        
        // Impact Section
        impactTitle: "您的影响",
        impactOpenSource: "开源",
        impactOpenSourceDesc: "我们所有的研究工具都免费提供给全球医疗社区使用",
        impactResearch: "学术研究",
        impactResearchDesc: "支持斯威本大学基于证据的医疗AI开发",
        impactGlobal: "全球影响",
        impactGlobalDesc: "创造改善全球医疗保健结果的技术",
        
        // Footer
        footerCopyright: "© 2024 医疗聊天机器人研究团队。保留所有权利。",
        footerInstitution: "斯威本科技大学AI研究倡议的一部分"
    }
};

// Function to update all UI strings based on selected language
function updateLanguage(lang) {
    currentLang = lang;
    const t = translations[lang];
    
    // Update navigation links (both desktop and mobile)
    const accountLinks = document.querySelectorAll('#nav-account, #nav-account-mobile');
    const subscriptionLinks = document.querySelectorAll('#nav-subscription, #nav-subscription-mobile');
    const aboutLinks = document.querySelectorAll('#nav-about, #nav-about-mobile');
    
    accountLinks.forEach(link => link.innerText = t.navAccount);
    subscriptionLinks.forEach(link => link.innerText = t.navSubscription);
    aboutLinks.forEach(link => link.innerText = t.navAbout);
    
    // Update hero section
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) heroTitle.innerText = t.heroTitle;
    
    const heroSubtitle = document.querySelector('.hero-subtitle');
    if (heroSubtitle) heroSubtitle.innerText = t.heroSubtitle;
    
    // Update mission section
    const missionTitle = document.querySelector('.mission-section h2');
    if (missionTitle) missionTitle.innerText = t.missionTitle;
    
    const missionTexts = document.querySelectorAll('.mission-text');
    if (missionTexts.length >= 4) {
        missionTexts[0].innerText = t.missionText1;
        missionTexts[1].innerText = t.missionText2;
        missionTexts[2].innerText = t.missionText3;
        missionTexts[3].innerText = t.missionHighlight;
    }
    
    const missionCta = document.querySelector('.mission-cta strong');
    if (missionCta) missionCta.innerText = t.missionCta;
    
    // Update donation section
    const donationTitle = document.querySelector('.donation-header h3');
    if (donationTitle) donationTitle.innerText = t.donationTitle;
    
    const donationSubtitle = document.querySelector('.donation-header p');
    if (donationSubtitle) donationSubtitle.innerText = t.donationSubtitle;
    
    // Update donation buttons
    const donationBtns = document.querySelectorAll('.donation-btn');
    if (donationBtns.length >= 4) {
        donationBtns[0].querySelector('small').innerText = t.coffeeAmount;
        donationBtns[1].querySelector('small').innerText = t.developmentAmount;
        donationBtns[2].querySelector('small').innerText = t.researchAmount;
        donationBtns[3].querySelector('small').innerText = t.livesAmount;
    }
    
    // Update custom donation
    const customLabel = document.querySelector('.custom-donation label');
    if (customLabel) customLabel.innerText = t.customAmountLabel;
    
    const customInput = document.querySelector('.custom-donation input');
    if (customInput) customInput.placeholder = t.customAmountPlaceholder;
    
    const customBtn = document.querySelector('.custom-donate-btn');
    if (customBtn) customBtn.innerHTML = `<i class="fas fa-heart"></i> ${t.donateButton}`;
    
    // Update donation footer
    const secureNote = document.querySelector('.donation-note');
    if (secureNote) secureNote.innerHTML = `<i class="fas fa-shield-alt"></i> ${t.secureNote}`;
    
    const researcherName = document.querySelector('.researcher-details h4');
    if (researcherName) researcherName.innerText = t.researcherName;
    
    const researcherTitle = document.querySelector('.researcher-details p');
    if (researcherTitle) researcherTitle.innerText = t.researcherTitle;
    
    const institution = document.querySelector('.institution');
    if (institution) institution.innerText = t.institution;
    
    // Update impact section
    const impactTitle = document.querySelector('.impact-section h2');
    if (impactTitle) impactTitle.innerText = t.impactTitle;
    
    const impactItems = document.querySelectorAll('.impact-item');
    if (impactItems.length >= 3) {
        impactItems[0].querySelector('h3').innerText = t.impactOpenSource;
        impactItems[0].querySelector('p').innerText = t.impactOpenSourceDesc;
        
        impactItems[1].querySelector('h3').innerText = t.impactResearch;
        impactItems[1].querySelector('p').innerText = t.impactResearchDesc;
        
        impactItems[2].querySelector('h3').innerText = t.impactGlobal;
        impactItems[2].querySelector('p').innerText = t.impactGlobalDesc;
    }
    
    // Update footer
    const footerTexts = document.querySelectorAll('.footer-content p');
    if (footerTexts.length >= 2) {
        footerTexts[0].innerText = t.footerCopyright;
        footerTexts[1].innerText = t.footerInstitution;
    }
}

// Function to toggle theme
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'dark';
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

// Function to initialize theme
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

// Mobile menu functions
function toggleMobileMenu() {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    hamburger.classList.toggle('active');
    mobileNav.classList.toggle('active');
}

function closeMobileMenu() {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    hamburger.classList.remove('active');
    mobileNav.classList.remove('active');
}

function isClickOutsideMobileMenu(event) {
    const hamburger = document.getElementById('hamburger-menu');
    const mobileNav = document.getElementById('mobile-nav');
    
    if (!hamburger || !mobileNav) return true;
    
    return !hamburger.contains(event.target) && !mobileNav.contains(event.target);
}

// Initialize mobile menu
function initializeMobileMenu() {
    // This function is called from the main script
}

// Redirect to Buy Me a Coffee with amount
function redirectToDonation(amount) {
    const baseUrl = 'https://buymeacoffee.com/kle1812';
    const url = amount ? `${baseUrl}?amount=${amount}` : baseUrl;
    window.open(url, '_blank');
}

// Initialize page when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Hide loading screen after animations complete
    setTimeout(() => {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.style.display = 'none';
        }
    }, 3000);
    
    // Initialize theme
    initializeTheme();
    
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    themeToggle.addEventListener('click', toggleTheme);
    
    // Language dropdown functionality
    const dropdownBtn = document.querySelector('.dropdown-btn');
    const dropdownMenu = document.querySelector('.dropdown-menu');
    
    dropdownBtn.addEventListener('click', function(event) {
        event.stopPropagation();
        dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
    });
    
    // Language selection
    document.querySelectorAll('.dropdown-menu li').forEach(item => {
        item.addEventListener('click', function(event) {
            event.stopPropagation();
            const selectedLang = this.getAttribute('data-lang');
            dropdownBtn.innerHTML = selectedLang + " &#x25BC;";
            dropdownMenu.style.display = 'none';
            updateLanguage(selectedLang);
        });
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function() {
        dropdownMenu.style.display = 'none';
    });
    
    // Donation button functionality
    document.querySelectorAll('.donation-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const amount = this.getAttribute('data-amount');
            redirectToDonation(amount);
        });
    });
    
    // Custom donation functionality
    const customDonateBtn = document.getElementById('custom-donate-btn');
    const customAmountInput = document.getElementById('custom-amount');
    
    customDonateBtn.addEventListener('click', function() {
        const amount = customAmountInput.value;
        if (amount && amount > 0) {
            redirectToDonation(amount);
        } else {
            alert('Please enter a valid amount');
        }
    });
    
    customAmountInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            customDonateBtn.click();
        }
    });
    
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
});
