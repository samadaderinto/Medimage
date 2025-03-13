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
import { loginSchema, LoginValues } from "@/lib/validation";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useForm } from "react-hook-form";
import Cookies from "js-cookie";

import HidePasswordIcon from "@/app/icons/HidePasswordIcon";
import Link from "next/link";
import { signInWithEmailAndPassword } from "firebase/auth";
import { useRouter } from "next/navigation";
import { auth } from "@/app/utils/firebase";
import toast from "react-hot-toast";
import ButtonSpinner from "@/app/components/ButtonSpinner";

const Page = () => {
  const router = useRouter();
  const form = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const [hidePassword, setHidePassword] = useState(true);

  const [loading, setLoading] = useState(false);
  //   const [isForgotPasswordOpen, setIsForgotPasswordOpen] = useState(false);

  const onSubmit = async (data: LoginValues) => {
    setLoading(true);
    signInWithEmailAndPassword(auth, data.email, data.password)
      .then(async (userCredential) => {
        const user = userCredential.user;
        Cookies.set("user_id", user?.uid);
        router.push("/dashboard");
        setLoading(false);
      })
      .catch((error) => {
        const errorMessage = error.message;
        toast.error(errorMessage);
        setLoading(false);
      });
  };

  return (
    <div className="size-full bg-[#F7F9FF] flex">
      <div className="flex-[2] flex justify-center items-center w-full h-full">
        <div className="flex flex-col w-1/2">
          <p className="text-4xl text-center font-medium">Welcome back</p>
          <p className="text-[#8495B8] text-center text-lg">
            Let&apos;s continue where we left
          </p>
          <div className="p-5 w-full my-5 bg-white rounded-2xl py-8">
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="w-full space-y-6"
              >
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
                          placeholder="Email here"
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
                <div className="flex justify-end w-full">
                  <p className="text-[#1358BD] text-sm hover:underline cursor-pointer">
                    Forgot password?
                  </p>
                </div>
                <Button className="w-full p-5 py-7 rounded-full" type="submit">
                  {loading ? <ButtonSpinner color="white" /> : "Login"}
                </Button>
                <div className="w-full justify-center items-center">
                  <p className="text-center">Don&apos;t have an account?</p>
                </div>
                <Link href={"/sign-up"}>
                  <Button
                    variant={"outline"}
                    className="w-full p-5 py-7 rounded-full"
                    type="button"
                  >
                    Create an account
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
