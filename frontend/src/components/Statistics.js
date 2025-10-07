import React from 'react';
import { useNavigate } from 'react-router-dom';
import Chart from 'react-apexcharts';
import './Statistics.css';

const Statistics = () => {
  const navigate = useNavigate();
  const epochs = Array.from({ length: 50 }, (_, i) => i);

  const lossOptions = {
    chart: { id: 'loss-chart', toolbar: { show: false } },
    xaxis: {
      categories: epochs,
      tickAmount: 10,
      title: { text: 'Epoch' },
    },
    colors: ['#000000', '#ec4755'],
    yaxis: { title: { text: 'Loss' } },
    title: { text: 'Training Loss Over Epochs', align: 'center' }
  };

  const lossSeries = [
    {
      name: 'Without Transformer',
      data: [6.2, 3.1, 2.5, 1.8, 1.4, 1.2, 1.1, 1.0, 0.95, 0.9, 0.88, 0.86, 0.84, 0.82, 0.81, 0.79, 0.78, 0.76, 0.75, 0.73, 0.72, 0.71, 0.7, 0.7, 0.7, 0.7, 0.69, 0.69, 0.69, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67]
    },
    { name: 'With Transformer', data: Array(50).fill(0.1) }
  ];

  const barOptions = {
    chart: { type: 'bar' },
    plotOptions: {
      bar: {
        distributed: true,
        borderRadius: 8,
        columnWidth: '50%'
      }
    },
    xaxis: {
      categories: ['Precision', 'Recall', 'F1-score']
    },
    title: { text: 'Model Evaluation Metrics', align: 'center' },
    dataLabels: {
      enabled: true,
      formatter: (val) => val.toFixed(4)
    },
    colors: ['#000000', '#808080','#ec4755'],
    yaxis: {
      max: 1.1,
      title: { text: 'Score' }
    }
  };

  const barSeries = [
    {
      name: 'Score',
      data: [0.7910, 0.9897, 0.8793]
    }
  ];

  return (
    <div className="statistics-background">
      <div className="statistics-container">
      <h2 className="result-title">
  <span style={{ color: '#1a2e35' }}>Model </span>
  <span style={{ color: '#ec4755' }}>Statistics</span>
      </h2>
        <p className="result-message">Training and Evaluation Summary</p>

        <div className="chart-row">
    <div>
      <Chart options={lossOptions} series={lossSeries} type="line" height={300} />
    </div>
    <div>
      <Chart options={barOptions} series={barSeries} type="bar" height={300} />
    </div>
  </div>
        <p className="chart-note"> We strive to reduce false positives for more accurate diagnoses.</p>

        <p className="navigation-note">
          To start prediction, head to{' '}
          <span className="home-link" onClick={() => navigate('/')}>Home</span>.
        </p>
      </div>
    </div>
  );
};

export default Statistics;


//Edit new stats 