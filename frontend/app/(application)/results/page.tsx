"use client";
import TopBar from "@/app/components/TopBar";
import WordByWordText from "@/app/components/WordByWordText";
import { IUploadProps } from "@/app/store/userStore";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { useRouter, useSearchParams } from "next/navigation";
import React from "react";

const Page = () => {
  const router = useRouter();
  const search = useSearchParams();
  const id = search.get("id");

  const demoResult: IUploadProps = {
    result: {
      image: {
        url: "",
        resolution: "1024 x 2048",
      },
      description:
        "This image shows an axial (horizontal cross-section) MRI scan of a brain. The scan appears to be T2-weighted, which is why the cerebrospinal fluid appears bright white. The image displays both hemispheres of the brain with clear visualization of structures like the ventricles (fluid-filled spaces) in the center. The brain tissue shows varying shades of gray, representing different tissue types and densities. The outer edges of the brain show the cortical folding pattern (gyri and sulci). There appears to be symmetry between the left and right hemispheres. The ventricles appear to be of normal size and shape. The brain parenchyma (tissue) shows normal signal intensity throughout without obvious focal areas of abnormal signal. The image quality is good with clear contrast between different brain structures.",
      diagnosis: "benign",
    },
    acccuracy: 87,
    patient: {
      firstName: "David",
      lastName: "Bajomo",
      email: "bajomodavid18@gmail.com",
      age: "21",
      condition: [],
    },
    timestamp: "2023-01-11T13:53:04.057Z",
  };

  const texts = {
    malignant:
      "Your test results indicate a malignant thyroid condition, which suggests the presence of thyroid cancer. It is crucial to consult an endocrinologist and an oncologist as soon as possible. Treatment options may include surgery, radioactive iodine therapy, targeted therapy, or radiation. Regular monitoring and follow-ups will be necessary to assess the response to treatment. Please schedule an appointment immediately for a personalized treatment plan.",

    benign:
      "Your test results indicate a benign thyroid condition. This means that while there may be nodules or growths, they are not cancerous. However, regular monitoring through ultrasound and thyroid function tests is recommended to ensure stability. If symptoms like difficulty swallowing, voice changes, or rapid growth occur, consult your doctor for further evaluation. In some cases, medication or minor procedures may be required for symptom management.",

    "normal thyroid":
      "Your test results indicate a normal and healthy thyroid function. No signs of abnormalities were detected. Continue maintaining a balanced diet rich in iodine, selenium, and zinc to support thyroid health. If you experience symptoms such as unexplained fatigue, weight changes, or neck swelling in the future, consult your doctor for further evaluation. Routine check-ups are always beneficial for long-term thyroid health.",
  };

  return (
    <div className="w-full h-full flex flex-col">
      <TopBar
        title={
          id || parseInt(id as string) === 0 ? "Result" : "Recent Activity"
        }
      />
      <div className="h-full overflow-hidden">
        {id || parseInt(id as string) === 0 ? (
          <div className="p-10 w-full h-full">
            <div className="flex justify-between gap-x-8">
              <div className="flex-[3] w-full p-5 rounded-2xl bg-white">
                <div className="w-full h-[300px] relative">
                  <Image src={"/scan1.png"} alt="scan" fill />
                </div>
                <div className="py-5">
                  <p className="text-lg font-semibold">AI Description</p>
                  <WordByWordText
                    text={demoResult.result?.description as string}
                  />
                </div>
              </div>
              <div className="flex-[2] bg-white p-5 rounded-2xl w-full">
                <p className="text-lg font-semibold pb-5">Analysis Result</p>
                <div className="my-1 text-[#4B5563]">
                  Patient Name:{" "}
                  <span className="font-semibold text-lg">
                    {demoResult?.patient.firstName}{" "}
                    {demoResult?.patient?.lastName}
                  </span>
                </div>
                <div className="my-1 text-[#4B5563]">
                  Age:{" "}
                  <span className="font-semibold text-lg">
                    {demoResult?.patient.age}
                  </span>
                </div>
                <div className="my-1 text-[#4B5563]">
                  Email:{" "}
                  <span className="font-semibold text-lg underline cursor-pointer">
                    {demoResult?.patient?.email}
                  </span>
                </div>
                <div className="my-1 text-[#4B5563]">
                  Prior conditions:{" "}
                  <span className="font-semibold text-lg">
                    {demoResult?.patient.condition.length < 1 ? (
                      "None"
                    ) : (
                      <span className="flex flex-wrap">
                        {demoResult?.patient?.condition?.map((item, index) => (
                          <span key={index} className="">
                            {item}
                          </span>
                        ))}
                      </span>
                    )}
                  </span>
                </div>
                <div className="my-1 text-[#4B5563]">
                  Diagnosis:{" "}
                  <span
                    className={cn(
                      "font-bold text-lg uppercase cursor-pointer",
                      {
                        "text-[#1CA556]":
                          demoResult?.result?.diagnosis === "normal thyroid",
                        "text-[#ffc406]":
                          demoResult?.result?.diagnosis === "benign",
                        "text-[#BD1E1E]":
                          demoResult?.result?.diagnosis === "malignant",
                      }
                    )}
                  >
                    {demoResult?.result?.diagnosis}
                  </span>
                </div>
                <div className="w-full py-2 pb-2 flex text-[#4B5563] justify-between">
                  <p>Diagnosis accuracy</p>
                  <p>{demoResult?.acccuracy}%</p>
                </div>
                <Progress
                  color={cn("", {
                    "bg-[#1CA556]":
                      demoResult?.result?.diagnosis === "normal thyroid",
                    "bg-[#ffc406]": demoResult?.result?.diagnosis === "benign",
                    "bg-[#BD1E1E]":
                      demoResult?.result?.diagnosis === "malignant",
                  })}
                  value={demoResult?.acccuracy}
                  className="h-1"
                />
                <div className="pt-10">
                  <p className="text-lg font-semibold">AI Advice:</p>
                  <WordByWordText
                    text={
                      texts[demoResult?.result?.diagnosis as keyof typeof texts]
                    }
                  />
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-4 p-10">
            {[0, 0, 0, 0, 0, 0].map((item, index) => {
              return (
                <div
                  onClick={() => router.push(`/results?id=${index}`)}
                  className="rounded-md border p-5 pb-2 w-full cursor-pointer"
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
        )}
      </div>
    </div>
  );
};

export default Page;
