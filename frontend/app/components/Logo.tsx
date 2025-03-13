"use client";
import Image from "next/image";
import React from "react";

const Logo = ({ width, height }: { width: number; height: number }) => {
  return (
    <Image
      src={"/logoImage.png"}
      className="object-cover"
      alt="logo"
      width={width}
      height={height}
    />
  );
};

export default Logo;
