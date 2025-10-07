import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import UploadPage from './components/UploadPage';
import ResultPage from './components/ResultPage';
import Statstics from './components/Statistics';
import GetStarted from './components/GetStarted';
import TextDetection from './components/TextDetection';
import TextOutput from './components/TextOutput';
import './styles.css';

const App = () => (
  <Router>
    <Routes>
    <Route path="/" element={<ExternalRedirect url="http://localhost:3000/" />} />
      <Route path="/text-detection" element={<TextDetection />} />
      <Route path="/text-output" element={<TextOutput />} />

      <Route path="/upload" element={<UploadPage />} />
      <Route path="/getstarted" element={<GetStarted />} />
      <Route path="/result" element={<ResultPage />} />
      <Route path="/statistics" element={<Statstics />} />
    </Routes>
  </Router>
);


function ExternalRedirect({ url }) {
  React.useEffect(() => {
    window.location.replace(url);
  }, [url]);

  return null;
}

export default App;
