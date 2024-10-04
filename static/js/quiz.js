// logic for quiz taking
async function loadQuiz() {
    const response = await fetch('/quiz');
    const data = await response.json();
    const quizContainer = document.getElementById('quiz-container');
    quizContainer.innerHTML = ''; 

    data.questions.forEach((q, index) => {
        const questionElement = document.createElement('div');
        questionElement.classList.add('mb-6'); 
        questionElement.innerHTML = `
            <h5 class="text-xl font-semibold text-gray-800">${q.question}</h5>
            <div class="mt-2">
                ${q.options.map(option => `
                    <label class="block mt-2">
                        <input type="radio" name="question${index}" value="${option}" class="mr-2">
                        <span class="text-gray-700">${option}</span>
                    </label>
                `).join('')}
            </div>
        `;
        quizContainer.appendChild(questionElement);
    });
}

// Submit the quiz and display result
async function submitQuiz() {
    const answers = [];
    document.querySelectorAll('input[type="radio"]:checked').forEach(input => {
        answers.push(input.value);
    });

    if (answers.length < 5) {
        alert("Please answer all questions.");
        return;
    }

    const response = await fetch('/quiz_result', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers })
    });
    const resultData = await response.json();
    document.getElementById('result').innerHTML = `<h3 class="text-xl text-gray-800">${resultData.result}</h3>`;
}

// Load quiz on page load
document.addEventListener('DOMContentLoaded', loadQuiz);

// Handle quiz submission
document.getElementById('submit-quiz').addEventListener('click', submitQuiz);