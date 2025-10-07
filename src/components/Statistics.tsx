import React from 'react';

const Statistics = () => {
  return (
    <section className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-center text-primary mb-8"></h1>
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <ul className="space-y-4 text-lg">
          <li>
            <strong>Accuracy:</strong>{' '}
          </li>
          <li>
            <strong>Precision:</strong>{' '}
          </li>
          <li>
            <strong>Recall:</strong>{' '}
          </li>
          <li>
            <strong>F1 Score:</strong>{' '}
          </li>
          <li>
            <strong>AUC:</strong>{' '}
          </li>
        </ul>
      </div>
    </section>
  );
};

export default Statistics;
