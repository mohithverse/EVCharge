const views = document.querySelectorAll('.view');
const navItems = document.querySelectorAll('[data-view]');
const title = document.getElementById('pageTitle');
const titles = { overview: 'Project overview', cleaning: 'Data cleaning report', eda: 'Exploratory data analysis', regression: 'Regression model results', gradient: 'Gradient descent results', 'tree-models': 'Tree model comparison' };
const toast = document.getElementById('toast');

function showView(id) {
  views.forEach(view => view.classList.toggle('active', view.id === id));
  document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item.dataset.view === id));
  title.textContent = titles[id];
  history.replaceState(null, '', `#${id}`);
  document.getElementById('sidebar').classList.remove('open');
  toast.textContent = `${titles[id]} loaded`;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 1600);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

navItems.forEach(item => item.addEventListener('click', () => showView(item.dataset.view)));
document.getElementById('menuButton').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));

document.querySelectorAll('.filter-tabs button').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.filter-tabs button').forEach(tab => tab.classList.remove('active'));
  button.classList.add('active');
  document.querySelectorAll('.chart-gallery figure').forEach(chart => chart.classList.toggle('hidden', button.dataset.filter !== 'all' && chart.dataset.category !== button.dataset.filter));
}));

document.getElementById('shuffleCharts').addEventListener('click', () => {
  const gallery = document.getElementById('chartGallery');
  [...gallery.children].sort(() => Math.random() - .5).forEach(card => gallery.appendChild(card));
  toast.textContent = 'Gallery refreshed'; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 1600);
});

const timeInput = document.getElementById('timeInput'), hourInput = document.getElementById('hourInput');
function updateEstimator() {
  const minutes = +timeInput.value, hour = +hourInput.value;
  // A calibrated, readable approximation of the trained two-feature model.
  const energy = Math.max(0.2, 0.121 * minutes + 0.062 * hour + 3.65);
  document.getElementById('timeValue').value = `${minutes} min`;
  document.getElementById('hourValue').value = `${String(hour).padStart(2, '0')}:00`;
  document.getElementById('estimateValue').innerHTML = `${energy.toFixed(1)} <b>kWh</b>`;
}
timeInput.addEventListener('input', updateEstimator); hourInput.addEventListener('input', updateEstimator); updateEstimator();

document.getElementById('themeButton').addEventListener('click', () => {
  document.body.classList.toggle('high-contrast');
  document.getElementById('themeButton').textContent = document.body.classList.contains('high-contrast') ? '☀' : '☾';
});

function formatMetric(value) {
  return value === null ? '—' : value.toFixed(3);
}

function renderModelResults(results) {
  const completeModels = results.models.filter(model => model.status === 'complete');
  const bestModel = [...completeModels].sort((a, b) => b.r2 - a.r2)[0];
  document.getElementById('modelStatus').textContent = `${results.dataset.test_rows.toLocaleString()} test rows`;
  document.getElementById('treeSummary').innerHTML = results.models.map(model => `
    <article class="metric-card model-card ${model.status !== 'complete' ? 'pending' : ''}">
      <p>${model.name}</p><strong>${model.r2 === null ? 'Pending' : `${(model.r2 * 100).toFixed(1)}%`}</strong>
      <small>${model.r2 === null ? 'install dependency to evaluate' : 'R² variance explained'}</small>
    </article>`).join('');
  document.getElementById('modelTable').innerHTML = results.models.map(model => `
    <tr class="${model.name === bestModel.name ? 'best-row' : ''}">
      <td><strong>${model.name}</strong>${model.name === bestModel.name ? '<span class="best-label">BEST R²</span>' : ''}</td>
      <td>${formatMetric(model.mae)}</td><td>${formatMetric(model.rmse)}</td><td>${formatMetric(model.r2)}</td><td>${model.concept}</td>
    </tr>`).join('');
  const sampleModel = results.models.find(model => model.samples);
  document.getElementById('sampleModelName').textContent = `${sampleModel.name} Regression`;
  document.getElementById('samplePredictions').innerHTML = sampleModel.samples.map(sample => {
    const error = sample.predicted - sample.actual;
    return `<tr><td>${sample.actual.toFixed(3)}</td><td>${sample.predicted.toFixed(3)}</td><td class="${error < 0 ? 'negative' : 'positive'}">${error >= 0 ? '+' : ''}${error.toFixed(3)}</td></tr>`;
  }).join('');
  const pending = results.models.find(model => model.status !== 'complete');
  document.getElementById('xgboostNote').innerHTML = pending ? `<strong>${pending.name} output pending:</strong> run <code>${pending.install}</code>, then update model_results.json by rerunning the model.` : 'All four models have complete test-set results.';
}

fetch('model_results.json')
  .then(response => response.json())
  .then(renderModelResults)
  .catch(() => {
    document.getElementById('modelStatus').textContent = 'Start with python -m http.server 8000';
  });

const initialView = location.hash.slice(1); if (titles[initialView]) showView(initialView);
