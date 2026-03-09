async function postPredict(payload) {
  const url = '/.netlify/functions/predict';
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  return res.json();
}

document.getElementById('predict-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const data = {};
  new FormData(form).forEach((v,k) => data[k] = v);
  // convert numeric fields
  Object.keys(data).forEach(k => { const n = Number(data[k]); if (!isNaN(n)) data[k] = n; });
  const out = document.getElementById('output');
  out.textContent = 'Predicting...';
  try {
    const r = await postPredict(data);
    if (r.error) {
      out.innerHTML = '<div class="text-danger">Error: ' + r.error + '</div>';
    } else {
      out.innerHTML = '<p><strong>Predicted default probability:</strong> ' + (r.predicted_default_probability).toFixed(6) + '</p>' +
                      '<p><strong>Risk category:</strong> ' + r.risk_category + '</p>';
    }
  } catch (err) {
    out.innerHTML = '<div class="text-danger">' + err + '</div>';
  }
});
