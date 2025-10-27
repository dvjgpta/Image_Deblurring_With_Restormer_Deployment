// Get references to HTML elements
const form = document.getElementById('uploadForm');
const loader = document.getElementById('loader');
const outputDiv = document.getElementById('output');

// Listen for the form submission event
form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent the default page reload
    loader.style.display = 'block'; // Show "Processing..."
    outputDiv.innerHTML = ''; // Clear previous output

    // Prepare the image file to send to the Flask backend
    const formData = new FormData(form);

    try {
        // Send POST request to Flask '/infer' route
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        // Parse JSON response
        const data = await response.json();
        loader.style.display = 'none'; // Hide loader once done

        // If inference was successful
        if (data.result) {
            outputDiv.innerHTML = `
                <h3>Deblurred Image</h3>
                <img src="${data.result}?${new Date().getTime()}" alt="Output Image">
                <br>
                <a href="${data.result}" download="deblurred.jpg">
                    <button>Download Result</button>
                </a>
            `;
        } else {
            // If the server returned an error message
            outputDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        }
    } catch (err) {
        loader.style.display = 'none';
        outputDiv.innerHTML = `<p style="color:red;">Something went wrong!</p>`;
        console.error(err);
    }
});
