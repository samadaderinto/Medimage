"use client";
import { useRouter } from "next/navigation";
import React, { useEffect } from "react";

const Page = () => {
  const router = useRouter();
  useEffect(() => {
    router.push("/login");
  }, [router]);
  return <div className="text-black">Page</div>;
};

export default Page;
