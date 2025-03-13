"use client";
import AccuracyChart from "@/app/components/AccuracyChart";
import TopBar from "@/app/components/TopBar";
import AccuracyIcon from "@/app/icons/AccuracyRate";
import SuccessfulUploadsIcon from "@/app/icons/SuccessfulUploadsIcon";
import UploadsIcon from "@/app/icons/UploadsIcon";
import { useUserDetails } from "@/app/store/userStore";
import Image from "next/image";
import React from "react";

const Page = () => {
  const userDetails = useUserDetails((state) => state.data);

  const getAccuracy = () => {
    const accuracies = userDetails?.upload?.map((item) => item.acccuracy);
    const averageAccuracy = accuracies?.reduce((sum, acc) => sum + acc, 0);
    return averageAccuracy;
  };

  const cards = [
    {
      title: "Total scans",
      icon: <UploadsIcon />,
      value: userDetails?.upload?.length || 0,
    },
    {
      title: "Successful scans",
      icon: <SuccessfulUploadsIcon />,
      value: userDetails?.upload?.length || 0,
    },
    // {
    //   title: "Total scans",
    //   icon: <UploadsIcon />,
    //   value: userDetails?.upload?.length || 0,
    // },
    {
      title: "Accuracy",
      icon: <AccuracyIcon />,
      value: getAccuracy() || 0,
    },
  ];

  return (
    <div className="w-full h-full flex flex-col">
      <TopBar title="Dashboard" />
      <div className="w-full h-full overflow-y-scroll p-8">
        <p className="text-xl font-semibold pb-4">
          Hello, {userDetails?.firstName} 👋🏾
        </p>
        <div className="h-full w-full space-y-5">
          <div className="grid grid-cols-3 gap-x-4">
            {cards.map((item) => {
              return (
                <div
                  className="p-5 rounded-md bg-white flex gap-x-4 w-full"
                  key={item.title}
                >
                  {item.icon}
                  <div className="flex flex-col">
                    <p className="text-[#4B5563] text-sm font-medium">
                      {item.title}
                    </p>
                    <p className="text-xl font-semibold">{item.value}</p>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="w-full p-5 rounded-md bg-white">
            <p className="text-lg font-semibold">Recent Activity</p>
            <div className="grid grid-cols-3 gap-4 pt-5">
              {[0, 0, 0].map((item, index) => {
                return (
                  <div
                    className="rounded-md border p-5 pb-2 w-full"
                    key={index}
                  >
                    <div className="w-full h-[150px] relative rounded-md">
                      <Image
                        alt="scan"
                        className="object-cover rounded-md"
                        src={"/scan1.png"}
                        fill
                      />
                    </div>
                    <div className="py-3 flex justify-between">
                      <div>
                        <p className="text-sm font-semibold">David Bajomo</p>
                        <p className="text-xs text-[#6B7280]">
                          Classified with 99% accuracy
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-[#6B7280] font-medium">
                          March 15, 2025
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="w-full bg-white p-5 rounded-md ">
            <p className="text-lg font-semibold pb-10">Accuracy Trend (%)</p>
            <AccuracyChart />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Page;
