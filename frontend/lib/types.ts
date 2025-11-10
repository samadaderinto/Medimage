export interface UserDetails {
  data?: {
    upload?: Array<{
      acccuracy: number;
      // Add other fields as needed
    }>;
  };
}

export interface UserStore {
  data: UserDetails | null;
  setData: (data: UserDetails) => void;
}