"use client";
import ButtonSpinner from "@/app/components/ButtonSpinner";
import TopBar from "@/app/components/TopBar";
import DownArrow from "@/app/icons/DownArrow";
import SelectedFilterIcon from "@/app/icons/SelectedFiltericon";
import UnselectedFilter from "@/app/icons/UnselectedFilterIcon";
import { Button } from "@/components/ui/button";
import { useDropzone } from "react-dropzone";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { patientDetailsSchema, PatientFormValues } from "@/lib/validation";
import { zodResolver } from "@hookform/resolvers/zod";
import React, { useCallback, useEffect, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import Image from "next/image";
import UploadFileIcon from "@/app/icons/UploadFileIcon";
import XIcon from "@/app/icons/XIcon";

const Page = () => {
  const thyroidConditions = [
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hashimoto's Thyroiditis",
    "Graves' Disease",
    "Goiter",
    "Thyroid Nodules",
    "Thyroid Cancer",
    "Postpartum Thyroiditis",
    "Subacute Thyroiditis",
    "Silent Thyroiditis",
    "Congenital Hypothyroidism",
    "Iodine Deficiency Hypothyroidism",
    "Euthyroid Sick Syndrome",
    "Toxic Multinodular Goiter",
    "Thyroid Storm",
    "Myxedema",
  ];

  const [thyroidArray, setThyroidArray] = useState<string[]>([]);

  const form = useForm<PatientFormValues>({
    resolver: zodResolver(patientDetailsSchema),
    defaultValues: {
      email: "",
      firstName: "",
      lastName: "",
      age: 0,
      conditions: [],
    },
  });

  const onSubmit = (values: PatientFormValues) => {
    console.log(values);
  };

  useEffect(() => {
    form.setValue("conditions", thyroidArray);
  }, [thyroidArray, form]);

  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        form.setValue("scan", file, { shouldValidate: true });

        // Generate preview
        setPreview(URL.createObjectURL(file));
      }
    },
    [form]
  );

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"] },
    maxSize: 5 * 1024 * 1024, // 5MB limit
  });

  return (
    <div className="w-full h-full flex flex-col">
      <TopBar title="Upload" />
      <div className="w-full h-full overflow-y-scroll p-8">
        <p className="text-3xl font-semibold">New Upload</p>
        <p>Upload your medical images for analysis and processing</p>
        <div className="p-10 mt-10 w-[60%] rounded-2xl bg-white">
          <p className="pb-8 font-bold text-lg">Patient Details</p>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="w-full space-y-5"
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

              <div className="flex gap-x-6 w-full">
                <FormField
                  control={form.control}
                  name="email"
                  render={({ field }) => (
                    <FormItem className="w-full">
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
                  name="age"
                  render={({ field }) => (
                    <FormItem className="w-full">
                      <FormLabel>Age</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          className="w-full bg-[#F7F9FF] border border-[#D6E2F9] px-3 py-5"
                          placeholder="Email"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="w-full mt-3">
                <div className="text-sm lg:text-base font-light flex gap-x-4 pb-2">
                  <FormLabel>Conditions</FormLabel>
                </div>
                <Popover>
                  <PopoverTrigger className={"outline-none w-full h-full "}>
                    <div
                      className={`${
                        form.formState.errors?.conditions
                          ? "border-red-600 border"
                          : ""
                      } relative min-h-12 max-h-max w-full flex-wrap overflow-x-scroll rounded-md bg-[#F7F9FF] border border-[#D6E2F9] flex gap-1 items-center justify-start outline-none font-sans text-sm text-[#818181]rounded-lg px-[1.05rem] py-2`}
                    >
                      {thyroidArray?.length > 0 ? (
                        thyroidArray?.map((item: string, index: number) => {
                          return (
                            <div
                              key={index}
                              className="bg-[#F0F6FF] p-1 text-xs items-center rounded-md flex gap-x-1"
                            >
                              <p className="text-neew-royal-blue capitalize">
                                {
                                  thyroidConditions.filter((i) => {
                                    return i === item || i === item;
                                  })[0]
                                }
                              </p>
                            </div>
                          );
                        })
                      ) : (
                        <p>Select Conditions</p>
                      )}
                      <div className="absolute right-4 top-1/2 -translate-y-1/2">
                        <DownArrow />
                      </div>
                    </div>
                  </PopoverTrigger>
                  <PopoverContent
                    className={
                      "w-[500px] h-[400px] overflow-y-scroll outline-none bg-white border shadow-lg text-sm p-4 rounded flex flex-col gap-y-2 transition-all z-10"
                    }
                  >
                    <div className="grid grid-cols-1 gap-1">
                      {thyroidConditions?.map((item: string, index: number) => {
                        const checked = thyroidArray?.includes(item);
                        return (
                          <div
                            key={index}
                            onClick={() => {
                              if (checked) {
                                setThyroidArray(
                                  thyroidArray?.filter(
                                    (thyCond) => thyCond !== item
                                  )
                                );
                              } else {
                                setThyroidArray([...thyroidArray, item]);
                              }
                            }}
                            className="flex gap-x-4 p-1"
                          >
                            {checked ? (
                              <SelectedFilterIcon />
                            ) : (
                              <UnselectedFilter />
                            )}
                            <p className="capitalize">{item}</p>
                          </div>
                        );
                      })}
                    </div>
                  </PopoverContent>
                </Popover>
                {form.formState.errors?.conditions && (
                  <p className="text-red-600 text-xs py-1">
                    {form.formState.errors?.conditions.message}
                  </p>
                )}
              </div>

              <Controller
                name="scan"
                control={form.control}
                render={() => (
                  <FormField
                    name="scan"
                    control={form.control}
                    render={() => (
                      <FormItem>
                        <FormLabel>Upload File</FormLabel>
                        <FormControl>
                          <div
                            {...getRootProps()}
                            className="border-dashed bg-[#F7F9FF] flex flex-col items-center border-2 border-[#D6E2F9] p-6 rounded-lg text-center cursor-pointer hover:bg-gray-100"
                          >
                            <input {...getInputProps()} />
                            <UploadFileIcon />
                            <p className="text-gray-600 pt-1">
                              Drag & drop a file here, or click to select one
                            </p>
                            {preview && (
                              <div className="w-full flex justify-center h-[300px] items-center relative">
                                <Image
                                  src={preview}
                                  alt="Preview"
                                  className="mt-2 w-24 h-24 rounded-lg object-contain"
                                  fill
                                />
                                <div
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    form.setValue("scan", undefined, {
                                      shouldValidate: true,
                                    });
                                    setPreview(null);
                                  }}
                                  className="absolute cursor-pointer top-0 right-0"
                                >
                                  <XIcon />
                                </div>
                              </div>
                            )}
                          </div>
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                )}
              />

              <Button
                className="w-full my-5 p-5 py-7 rounded-full"
                type="submit"
              >
                {false ? <ButtonSpinner color="white" /> : "Submit"}
              </Button>
            </form>
          </Form>
        </div>
      </div>
    </div>
  );
};

export default Page;
