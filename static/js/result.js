// logic for fetching model information
document.getElementById('fetch-info').addEventListener('click', function () {
    fetch('/model_info')
        .then(response => response.text())
        .then(data => {
            document.getElementById('model-info').textContent = data;
        })
        .catch(error => {
            console.error('Error fetching model info:', error);
        });
});

document.getElementById('load-history').addEventListener('click', function () {
    fetch('/prediction_history')
        .then(response => response.json())
        .then(data => {
            const history = data.history;  // Access 'history' array from the JSON response
            const tbody = document.getElementById('history-body');
            tbody.innerHTML = ''; 
            
            history.forEach((item, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="py-2 px-4 border-b">${index + 1}</td>
                    <td class="py-2 px-4 border-b">${JSON.stringify(item)}</td>
                `;
                tbody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Error fetching prediction history:', error);
        });
});