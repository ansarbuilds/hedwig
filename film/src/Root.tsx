import React from "react";
import { Composition } from "remotion";
import { HedwigFilm, HedwigHeroLoop } from "./HedwigFilm";

export const RemotionRoot: React.FC = () => {
  return (
    <>
    <Composition id="HedwigHeroLoop" component={HedwigHeroLoop} durationInFrames={180} fps={30} width={1920} height={1080} />
    <Composition
      id="HedwigDemo"
      component={HedwigFilm}
      durationInFrames={1800}
      fps={30}
      width={1920}
      height={1080}
    />
    </>
  );
};
