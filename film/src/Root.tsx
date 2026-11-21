import React from "react";
import { Composition } from "remotion";
import { HedwigFilm } from "./HedwigFilm";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="HedwigDemo"
      component={HedwigFilm}
      durationInFrames={1800}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
