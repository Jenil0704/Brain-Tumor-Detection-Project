document.getElementById('appointmentForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);

    try {
        const response = await fetch('/api/book_appointment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            document.querySelector('.success-message').classList.remove('hidden');
            document.querySelector('.error-message').classList.add('hidden');
            this.reset();
        } else {
            document.querySelector('.success-message').classList.add('hidden');
            document.querySelector('.error-message').classList.remove('hidden');
        }
    } catch (error) {
        document.querySelector('.success-message').classList.add('hidden');
        document.querySelector('.error-message').classList.remove('hidden');
    }
});