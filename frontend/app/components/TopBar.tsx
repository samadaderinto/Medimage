"use client";
import { collection, doc, getDoc } from "firebase/firestore";
import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
import { myDb } from "../utils/firebase";
import { getAuth, signOut } from "firebase/auth";
import Cookies from "js-cookie";
import toast from "react-hot-toast";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import LogoutIcon from "../icons/LogoutIcon";
import { getInitials } from "@/lib/strings";

const TopBar = ({ title }: { title: string }) => {
  const router = useRouter();
  const [details, setDetails] = useState<{
    firstName: string;
    lastName: string;
    email: string;
  }>({
    firstName: "",
    lastName: "",
    email: "",
  });

  const getUserProfile = async (id: string) => {
    const querySnapshot = await getDoc(doc(collection(myDb, "users"), id));
    if (querySnapshot) {
      setDetails({
        firstName: querySnapshot.data()?.firstName as string,
        lastName: querySnapshot.data()?.lastName as string,
        email: querySnapshot.data()?.email as string,
      });
    }
  };

  const logout = async () => {
    const auth = getAuth();
    signOut(auth)
      .then(() => {
        Cookies.remove("user_id");
        router.push("/login");
      })
      .catch((error) => {
        toast.error(error);
      });
  };

  useEffect(() => {
    getUserProfile(Cookies.get("user_id") as string);
  }, []);
  return (
    <Popover>
      <div className="h-[80px] bg-white p-5 border-b flex justify-between items-center">
        <div className="text-lg font-semibold">{title}</div>
        <PopoverTrigger>
          <div className="flex gap-x-4 items-center">
            <div className="bg-[#2563EB] text-white w-10 h-10 font-semibold rounded-full flex justify-center items-center">
              {getInitials(`${details.firstName} ${details.lastName}`)}
            </div>
            <div className="text-left pr-8">
              <p>
                {details.firstName} {details.lastName}
              </p>
              <p className="test-sm text-xs text-[#8495B8]">
                Monday 15 March, 2025
              </p>
            </div>
          </div>
        </PopoverTrigger>
        <PopoverContent>
          <div
            onClick={logout}
            className="cursor-pointer flex gap-x-2 items-center p-2"
          >
            <LogoutIcon />
            <p className="text-xs sm:text-sm font-light text-[#FF0000]">
              Log Out
            </p>
          </div>
        </PopoverContent>
      </div>
    </Popover>
  );
};

export default TopBar;
