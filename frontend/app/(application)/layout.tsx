"use client";
import React, { useEffect } from "react";
import SideBar from "../components/SideBar";
import Cookies from "js-cookie";
import { collection, doc, getDoc } from "firebase/firestore";
import { myDb } from "../utils/firebase";
import { useUserDetails } from "../store/userStore";

const Layout = ({ children }: { children: React.ReactNode }) => {
  const setUserDetails = useUserDetails((state) => state.setUserDetails);

  const getUserProfile = async (id: string) => {
    const querySnapshot = await getDoc(doc(collection(myDb, "users"), id));
    if (querySnapshot) {
      setUserDetails({
        firstName: querySnapshot.data()?.firstName as string,
        lastName: querySnapshot.data()?.lastName as string,
        email: querySnapshot.data()?.email as string,
        upload: querySnapshot.data()?.uploads,
      });
    }
  };

  useEffect(() => {
    getUserProfile(Cookies.get("user_id") as string);
  }, []);

  return (
    <div className="flex w-screen h-screen overflow-hidden bg-[#F7F9FF]">
      <div className="">
        <SideBar />
      </div>
      <div className="w-full">{children}</div>
    </div>
  );
};

export default Layout;
