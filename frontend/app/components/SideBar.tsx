"use client";
import React from "react";
import { DashboardActive, DashboardBase } from "../icons/DashboardIcon";
import { UploadActive, UploadBase } from "../icons/UploadIcon";
import { ResultsActive, ResultsBase } from "../icons/Results";
import Logo from "./Logo";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const SideBar = () => {
  const links = [
    {
      id: 1,
      title: "Dashboard",
      path: "/dashboard",
      activeIcon: <DashboardActive />,
      baseIcon: <DashboardBase />,
    },
    {
      id: 2,
      title: "Upload",
      path: "/upload",
      activeIcon: <UploadActive />,
      baseIcon: <UploadBase />,
    },
    {
      id: 3,
      title: "Results",
      path: "/results",
      activeIcon: <ResultsActive />,
      baseIcon: <ResultsBase />,
    },
  ];

  const pathname = usePathname();
  return (
    <div className="w-[250px] h-screen bg-white border-r p-5">
      <div className="py-8 w-full flex justify-center">
        <Logo width={100} height={100} />
      </div>
      <div className="flex flex-col gap-y-4">
        {links?.map((item) => {
          const active = pathname === item.path;
          return (
            <Link href={item.path} key={item.id}>
              <div className="flex gap-x-4 py-2">
                {active ? item.activeIcon : item.baseIcon}
                <p className={cn("", { "font-semibold": active })}>
                  {item.title}
                </p>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
};

export default SideBar;
