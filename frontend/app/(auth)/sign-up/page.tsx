"use client";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import React, { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { SignUpFormValues, signUpSchema } from "@/lib/validation";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useForm } from "react-hook-form";

import HidePasswordIcon from "@/app/icons/HidePasswordIcon";
import Link from "next/link";
import { createUserWithEmailAndPassword, getAuth } from "firebase/auth";
import { myApp, myDb } from "@/app/utils/firebase";
import Cookies from "js-cookie";
import { collection, doc, getDoc, setDoc } from "firebase/firestore";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import ButtonSpinner from "@/app/components/ButtonSpinner";

const Page = () => {
  const router = useRouter();
  const form = useForm<SignUpFormValues>({
    resolver: zodResolver(signUpSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const [hidePassword, setHidePassword] = useState(true);
  const [loading, setLoading] = useState(false);

  const checkUserExists = async (id: string) => {
    const querySnapshot = await getDoc(doc(collection(myDb, "users"), id));

    return querySnapshot?.exists();
  };
  const creatUserProfile = async (
    id: string,
    firstName: string,
    lastName: string,
    email: string
  ) => {
    await setDoc(doc(collection(myDb, "users"), id), {
      firstName,
      lastName,
      email,
    });
    // await addDoc(collection(myDb, "users", id), {
    //   firstName,
    //   lastName,
    // });
    return;
  };

  const onSubmit = async (data: SignUpFormValues) => {
    setLoading(true);
    const auth = getAuth(myApp);

    createUserWithEmailAndPassword(auth, data.email, data.password)
      .then(async (userCredential) => {
        const user = userCredential.user;
        Cookies.set("user_id", user?.uid);
        let userExists = await checkUserExists(user?.uid);
        if (!userExists) {
          await creatUserProfile(
            user?.uid,
            data?.firstName,
            data?.lastName,
            data?.email
          );
        }
        userExists = await checkUserExists(user?.uid);
        if (userExists) {
          Cookies.set("user_id", user?.uid);
          router.push("/dashboard");
        } else {
          toast.error("User could not be created");
        }
        setLoading(false);
      })
      .catch((error) => {
        // const errorCode = error.code;
        const errorMessage = error.message;
        toast.error(errorMessage);
        setLoading(false);
        // ..
      });
  };

  return (
    <div className="size-full bg-[#F7F9FF] flex">
      <div className="flex-[2] flex justify-center items-center w-full h-full">
        <div className="flex flex-col w-1/2">
          <p className="text-4xl text-center font-medium">Hello there, 👋🏾</p>
          <p className="text-[#8495B8] text-center text-lg">
            Let&apos;s get you started
          </p>
          <div className="p-5 w-full my-5 bg-white rounded-2xl py-8">
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="w-full space-y-4"
              >
                <div className="flex gap-x-6 w-full">
                  <FormField
                    control={form.control}
                    name="firstName"
                    render={({ field }) => (
                      <FormItem className="w-full">
                        <FormLabel>First Name</FormLabel>
                        <FormControl>
                          <Input
                            type="text"
                            className="w-full bg-[#F7F9FF] border border-[#D6E2F9] px-3 py-5"
                            placeholder="John"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="lastName"
                    render={({ field }) => (
                      <FormItem className="w-full">
                        <FormLabel>Last Name</FormLabel>
                        <FormControl>
                          <Input
                            type="text"
                            className="w-full bg-[#F7F9FF] border border-[#D6E2F9] px-3 py-5"
                            placeholder="Doe"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="email"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Email address</FormLabel>
                      <FormControl>
                        <Input
                          type="email"
                          className="w-full bg-[#F7F9FF] border border-[#D6E2F9] px-3 py-5"
                          placeholder="Email"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="password"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Password</FormLabel>
                      <FormControl>
                        <div className="w-full relative">
                          <Input
                            type={hidePassword ? "password" : "text"}
                            className="w-full bg-[#F7F9FF] border border-[#D6E2F9] px-3 py-5"
                            placeholder="Password"
                            {...field}
                          />
                          <button
                            type="button"
                            onClick={() => setHidePassword(!hidePassword)}
                            className="absolute right-2 top-1/2 -translate-y-1/2"
                          >
                            <HidePasswordIcon />
                          </button>
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="confirmPassword"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Confirm Password</FormLabel>
                      <FormControl>
                        <div className="w-full relative">
                          <Input
                            type={hidePassword ? "password" : "text"}
                            className="w-full bg-[#F7F9FF] border border-[#D6E2F9] px-3 py-5"
                            placeholder="Confirm Password"
                            {...field}
                          />
                          <button
                            type="button"
                            onClick={() => setHidePassword(!hidePassword)}
                            className="absolute right-2 top-1/2 -translate-y-1/2"
                          >
                            <HidePasswordIcon />
                          </button>
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button className="w-full p-5 py-7 rounded-full" type="submit">
                  {loading ? <ButtonSpinner color="white" /> : "Submit"}
                </Button>
                <div className="w-full justify-center items-center">
                  <p className="text-center text-sm">
                    Already have an account?
                  </p>
                </div>
                <Link href={"/login"}>
                  <Button
                    variant={"outline"}
                    className="w-full p-5 py-7 rounded-full"
                    type="button"
                  >
                    Login
                  </Button>
                </Link>
              </form>
            </Form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Page;
