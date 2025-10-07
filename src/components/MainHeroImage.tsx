import React from 'react';

import config from '../config/index.json';

const MainHeroImage = () => {
  const { mainHero } = config;
  return (
    <div className="lg:absolute lg:inset-y-0 lg:right-0 lg:w-1/2 flex justify-center items-center">
      <img
        className="h-[420px] w-[560px] rounded-3xl object-cover"
        src={mainHero.img}
        alt="hero image"
      />
    </div>
  );
};

export default MainHeroImage;
