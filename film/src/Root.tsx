import React from "react";
import { Composition } from "remotion";
import { HedwigDemo } from "./HedwigDemo";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="HedwigDemo"
      component={HedwigDemo}
      durationInFrames={1800}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
