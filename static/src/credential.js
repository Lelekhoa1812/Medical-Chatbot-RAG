document.addEventListener("DOMContentLoaded", () => {
    const loginRegisterModal = document.getElementById("loginRegisterModal");
    // Show the credential modal on "Account" nav click
    const navAccountLink = document.getElementById("nav-account");
    navAccountLink.addEventListener("click", (event) => {
        event.preventDefault(); // Prevent any default link navigation
        loginRegisterModal.style.display = "flex"; // or "block"
    });
    // Close modal on outside click
    loginRegisterModal.addEventListener("click", (event) => {
        // If user clicked directly on the overlay (not the inner wrapper)
        if (event.target === loginRegisterModal) {
        loginRegisterModal.style.display = "none";
        }
    });
    
    // Define login/register sections
    const loginSection = document.getElementById("loginSection");
    const registerSection = document.getElementById("registerSection");
    // Toggle showing between section and collapse other
    const toggleSection = (showLogin) => {
        if (showLogin) {
            loginSection.classList.remove("collapsed");
            registerSection.classList.add("collapsed");
        } else {
            loginSection.classList.add("collapsed");
            registerSection.classList.remove("collapsed");
        }
    };

    // On toggle per which section
    document.getElementById("showRegisterBtn").addEventListener("click", () => toggleSection(false));
    document.getElementById("showLoginBtn").addEventListener("click", () => toggleSection(true));

    // Login
    document.getElementById("loginBtn").addEventListener("click", async () => {
        const userName = document.getElementById("userName").value.trim();
        const userPassword = document.getElementById("userPassword").value.trim();
        // Mandatory fields for login
        if (!userName || !userPassword) {
            alert("Please fill in both fields.");
            return;
        }
        if (userPassword.length < 8) {
            alert("Password must be at least 8 characters long.");
            return;
        }
        // Connect to DB and post user login credential data as JSON
        try {
            const response = await fetch(`${BASE_URL}/user/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_name: userName, user_password: userPassword }),
            });
            // Attempt to match credential information
            if (response.ok) {
                    const loginRegisterModal = document.getElementById("loginRegisterModal");
                    loginRegisterModal.style.display = "none"; // Hide modal when login successfully
                } else {
                alert("Invalid login credentials.");
            }
        } catch (err) {
            console.error(err);
            alert("Server error!");
        }
    });
    // Register
    document.getElementById("createAccountBtn").addEventListener("click", async () => {
        const newUserName = document.getElementById("newUserName").value.trim();
        const newUserPassword = document.getElementById("newUserPassword").value.trim();        
        // Make sure data is valid
        if (!newUserName || !newUserPassword || !newUserPassword.length) {
            alert("Please provide a username and a password.");
            return;
        }
        // Enforce at least 8 chars
        if (newUserPassword.length < 8) {
            alert("Password must be at least 8 characters long.");
            return;
        }
        // Connect to DB and register new account
        try {
            const response = await fetch(`${BASE_URL}/user/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_name: newUserName, user_password: newUserPassword }),
            });
            // Catch account creation
            if (response.ok) {
                alert("Account created successfully! You can now log in.");
                toggleSection(true);
            // Cannot having duplicated user name in account
            } else {
                alert("Failed to create account. Username might already exist.");
            }
        // Cannot create account because DB or connection error
        } catch (err) {
            console.error(err);
            alert("Error creating account. Please try again.");
        }
    });  

    // Show login section by default
    toggleSection(true);
});