import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export interface IUserStoreProps {
  data: IUserProps | null;
  setUserDetails: (payload: IUserProps | null) => void;
}

export interface IPatientProps {
  firstName: string;
  lastName: string;
  age: string;
  email: string;
  condition: string[];
}

export interface ImageProps {
  url: string; //cloudinary image link from group 3
  resolution: string; // e.g 1093 x 5084  also from group 3
}

export interface ResultProps {
  image: ImageProps;
  description: string; // repo 2 response
  diagnosis: "normal thyroid" | "malignant" | "benign"; // group 4 response diagnosis
}

export interface IUploadProps {
  patient: IPatientProps;
  result?: ResultProps;
  acccuracy: number;
  timestamp: string;
}

export interface IUserProps {
  firstName: string;
  lastName: string;
  email: string;
  upload?: IUploadProps[];
}

export const useUserDetails = create<IUserStoreProps>()(
  persist(
    (set) => ({
      data: null,
      setUserDetails(payload) {
        set(() => ({ data: payload }));
      },
    }),
    {
      name: "userDetails",
      storage: createJSONStorage(() => sessionStorage),
    }
  )
);
