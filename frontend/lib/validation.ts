import z from "zod";

export const passwordComplexityRegex =
  /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z0-9]).{8,}$/;

const loginSchema = z.object({
  email: z.string({ required_error: "This field is required" }).email(),
  password: z.string({ required_error: "This field is required" }),
});

type LoginValues = z.infer<typeof loginSchema>;

const signUpSchema = z
  .object({
    confirmPassword: z
      .string({ required_error: "This field is required" })
      .min(1, { message: "This field is required" }),
    email: z.string({ required_error: "This field is required" }).email(),
    firstName: z
      .string({ required_error: "This field is required" })
      .min(1, { message: "First name is required" }),
    lastName: z
      .string({ required_error: "This field is required" })
      .min(1, { message: "Last name is required" }),
    password: z
      .string({ required_error: "This field is required" })
      .min(8, { message: "Password must contain at least 8 characters" })
      .regex(passwordComplexityRegex, {
        message:
          "Password must include uppercase, lowercase, digit, and special character.",
      }),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

type SignUpFormValues = z.infer<typeof signUpSchema>;

const fileSchema = z
  .instanceof(File)
  .refine((file) => file.size < 5 * 1024 * 1024, "File size must be <5MB")
  .refine(
    (file) => ["image/png", "image/jpeg"].includes(file.type),
    "Only PNG/JPEG allowed"
  );

const patientDetailsSchema = z.object({
  email: z.string({ required_error: "This field is required" }).email(),
  firstName: z
    .string({ required_error: "This field is required" })
    .min(1, { message: "First name is required" }),

  lastName: z.string({ required_error: "This field is required" }),
  age: z.number({ required_error: "This field is required" }),
  conditions: z
    .array(z.string())
    .min(2, "At least 2 items required")
    .max(5, "No more than 5 items allowed"),
  scan: fileSchema.optional(),
});

type PatientFormValues = z.infer<typeof patientDetailsSchema>;

export {
  loginSchema,
  type LoginValues,
  signUpSchema,
  type SignUpFormValues,
  patientDetailsSchema,
  type PatientFormValues,
};
