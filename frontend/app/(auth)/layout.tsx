"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import React from "react";
import Logo from "../components/Logo";

const OnboardlingLayout = ({ children }: { children: React.ReactNode }) => {
  const router = useRouter();

  return (
    <div className="w-screen h-screen flex text-black">
      <div className="flex-[3] flex flex-col">
        <div
          onClick={() => router.push("/")}
          className="w-full flex gap-x-4 bg-[#F7F9FF] cursor-pointer h-max justify-center items-center font-semibold text-xl px-8 py-5"
        >
          <Logo width={200} height={200} />
        </div>
        <div className="h-full overflow-y-scroll no-scrollbar">{children}</div>
      </div>

      <div className="hidden md:block p-4 bg-[#FAFAFA] flex-[2]">
        <div className="bg-[#FAFAFA] relative w-full h-full">
          <div className="bg-[#2948a9cc] absolute p-10 flex justify-center flex-col w-full h-full items-center text-white z-20">
            <p className="text-4xl font-semibold">
              Welcome to UNILAG Medical Imaging
            </p>
            <p className="text-lg">
              Access your medical imaging platform for advanced diagnostics and
              patient care management
            </p>
            <div className="absolute bottom-5 left-5">
              <p className="font-semibold">BME Class of 24&apos;</p>
            </div>
          </div>
          <Image
            alt="a black entrepreneur"
            className="object-cover rounded-lg z-10"
            src={"/authImage.png"}
            fill
          />
        </div>
      </div>
    </div>
  );
};

export default OnboardlingLayout;
